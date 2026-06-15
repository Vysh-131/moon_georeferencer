import concurrent.futures
import os
import shutil
import subprocess
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Default Projection (Lunar WAC Equirectangular)
# -----------------------------------------------------------------------------
DEFAULT_ISIS_MAP = """Group = Mapping
  ProjectionName     = Equirectangular
  CenterLongitude    = 0.0
  CenterLatitude     = 0.0
  TargetName         = Moon
  EquatorialRadius   = 1737400.0 <meters>
  PolarRadius        = 1737400.0 <meters>
  LatitudeType       = Planetocentric
  LongitudeDirection = PositiveEast
  LongitudeDomain    = 180
End_Group"""

DEFAULT_GDAL_WKT = (
    'PROJCRS["SimpleCylindrical_MOON",BASEGEOGCRS["GCS_MOON",DATUM["MOON",'
    'ELLIPSOID["MOON",1737400,0,LENGTHUNIT["metre",1,ID["EPSG",9001]]]],'
    'PRIMEM["Reference_Meridian",0,ANGLEUNIT["degree",0.0174532925199433,ID["EPSG",9122]]]],'
    'CONVERSION["Equidistant Cylindrical",METHOD["Equidistant Cylindrical",ID["EPSG",1028]],'
    'PARAMETER["Latitude of 1st standard parallel",0,ANGLEUNIT["degree",0.0174532925199433],'
    'ID["EPSG",8823]],PARAMETER["Longitude of natural origin",0,ANGLEUNIT["degree",0.0174532925199433],'
    'ID["EPSG",8802]],PARAMETER["False easting",0,LENGTHUNIT["metre",1],ID["EPSG",8806]],'
    'PARAMETER["False northing",0,LENGTHUNIT["metre",1],ID["EPSG",8807]]],CS[Cartesian,2],'
    'AXIS["easting",east,ORDER[1],LENGTHUNIT["metre",1,ID["EPSG",9001]]],'
    'AXIS["northing",north,ORDER[2],LENGTHUNIT["metre",1,ID["EPSG",9001]]]]'
)

def get_isis_binary(command_name):
    # Locates the required ISIS or GDAL binary in the system path.
    binary_path = shutil.which(command_name)
    if binary_path is None:
        if command_name.startswith("gdal"):
            return command_name 
        print(f"CRITICAL ERROR: Cannot find '{command_name}'. Ensure ISIS is activated.")
        sys.exit(1)
    return binary_path

def run_command(cmd, description):
    # Wrapper to execute system commands safely and capture output.
    cmd[0] = get_isis_binary(cmd[0])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] {description}")
        print(f"Error output:\n{e.stderr}")
        return False

def setup_templates(output_dir, custom_map=None, custom_wkt=None):
    # Sets up the mapping and WKT templates.
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Generate defaults
    map_file = Path(custom_map) if custom_map else out_path / "global_equatorial.map"
    wkt_file = Path(custom_wkt) if custom_wkt else out_path / "global_wac_crs.wkt"
    
    if not custom_map:
        with open(map_file, 'w') as f:
            f.write(DEFAULT_ISIS_MAP)
            
    if not custom_wkt:
        with open(wkt_file, 'w') as f:
            f.write(DEFAULT_GDAL_WKT)
            
    return map_file, wkt_file

def process_single_image(input_path, output_dir, map_template, crs_wkt_file):
    # Processes directory
    input_file = Path(input_path)
    output_folder = Path(output_dir)
    
    raw_cub = output_folder / f"{input_file.stem}.cub"
    mapped_cub = output_folder / f"{input_file.stem}_mapped.cub"
    final_output = output_folder / f"g_{input_file.stem}.tif" 
    
    if final_output.exists():
        return f"Skipped (Already processed): {final_output.name}"

    print(f"Processing: {input_file.name}...")
    status = f"Success: {final_output.name}"

    try:
        if not run_command(["lronac2isis", f"from={input_file}", f"to={raw_cub}"], f"Ingest {input_file.name}"): 
            return f"Failed at Ingestion: {input_file.name}"

        if not run_command(["spiceinit", f"from={raw_cub}", "web=yes", "shape=ellipsoid"], f"Spiceinit {input_file.name}"): 
            return f"Failed at Spiceinit: {input_file.name}"

        # Map to the global template but preserve camera footprint and resolution
        cmd_map = [
            "cam2map",
            f"from={raw_cub}",
            f"to={mapped_cub}",
            f"map={map_template}", 
            "pixres=camera",         
            "defaultrange=camera"  
        ]
        if not run_command(cmd_map, f"Cam2Map {input_file.name}"): 
            return f"Failed at Orthorectification: {input_file.name}"

        # Hard-inject the target WKT string
        cmd_export = [
            "gdal_translate",
            "-of", "GTiff",
            "-a_srs", str(crs_wkt_file),
            str(mapped_cub),
            str(final_output)
        ]
        if not run_command(cmd_export, f"GDAL {input_file.name}"): 
            return f"Failed at Export: {input_file.name}"

    finally:
        # Cleanup intermediate ISIS cubes to save disk space
        if raw_cub.exists(): 
            raw_cub.unlink()
        if mapped_cub.exists(): 
            mapped_cub.unlink()

    return status

def batch_process(input_folder, output_folder, max_workers, custom_map=None, custom_wkt=None):
    """Manages the parallel processing of all images in the input directory."""
    in_dir = Path(input_folder)
    out_dir = Path(output_folder)
    
    if not in_dir.exists() or not in_dir.is_dir():
        print(f"CRITICAL ERROR: Input directory '{in_dir}' does not exist.")
        sys.exit(1)

    image_files = list(in_dir.glob('*.IMG')) + list(in_dir.glob('*.img'))
    total_images = len(image_files)
    
    if total_images == 0:
        print(f"No valid .IMG files found in {in_dir}")
        return

    print(f"Found {total_images} images.")
    print(f"Processing with {max_workers} concurrent workers.")
    print("-" * 60)

    # Setup alignment templates
    map_template, crs_wkt_file = setup_templates(out_dir, custom_map, custom_wkt)

    success_count = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_image, img, out_dir, map_template, crs_wkt_file): img 
            for img in image_files
        }
        
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result and "Success" in result: 
                success_count += 1
            elif result and "Skipped" not in result: 
                print(result)

    print("-" * 60)
    print(f"Batch Complete: {success_count}/{total_images} images successfully processed.")

def main():
    print("=" * 60)
    print("LRO NAC Image Batch Processor")
    print("=" * 60)

    # --- Input Directory ---
    input_dir = input("Enter path to raw .IMG files (Default: ./raw_images): ").strip()
    if not input_dir:
        input_dir = "./raw_images"

    # --- Output Directory ---
    output_dir = input("Enter path for processed files (Default: ./processed_images): ").strip()
    if not output_dir:
        output_dir = "./processed_images"

    # --- Worker Count ---
    default_workers = max(1, (os.cpu_count() or 2) - 1)
    workers_input = input(f"Enter number of parallel workers (Default: {default_workers}): ").strip()
    
    if not workers_input:
        workers = default_workers
    else:
        try:
            workers = int(workers_input)
        except ValueError:
            print(f"Invalid input for workers. Defaulting to {default_workers}.")
            workers = default_workers

    # --- Custom Map/WKT (Optional) ---
    custom_map = input("Enter path to custom .map file (Leave blank for Moon WAC default): ").strip()
    if not custom_map:
        custom_map = None

    custom_wkt = input("Enter path to custom .wkt file (Leave blank for Moon WAC default): ").strip()
    if not custom_wkt:
        custom_wkt = None

    print("\nStarting process pipeline...\n" + "-" * 60)
    
    batch_process(
        input_folder=input_dir, 
        output_folder=output_dir, 
        max_workers=workers,
        custom_map=custom_map,
        custom_wkt=custom_wkt
    )

if __name__ == "__main__":
    main()
