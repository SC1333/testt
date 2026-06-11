from enum import Enum
from pathlib import Path
from typing import Dict, Tuple, Any
import csv
import time
import pandas as pd
import numpy as np

# --- Definitions & Constants ---

SEPARATOR_PREFIX = "==============="

class BikeType(Enum):
    NOT_CHOSEN = 0
    CITY_TREKKING_BIKE = 1
    ROAD_RACING_BIKE = 2
    E_BIKE = 3
    RECUMBENT_BICYCLE = 4
    FREIGHT_BICYCLE = 5
    TANDEM_BICYCLE = 6
    MOUNTAINBIKE = 7
    OTHER = 8

class PhoneLocation(Enum):
    POCKET = 0
    HANDLEBAR = 1
    JACKET_POCKET = 2
    HAND = 3
    BASKET_PANNIER = 4
    BACKPACK_BAG = 5
    OTHER = 6

class IncidentType(Enum):
    DUMMY_INCIDENT = -5
    NOTHING = 0
    CLOSE_PASS = 1
    PULLING_IN_OR_OUT = 2
    NEAR_HOOK = 3
    APPROACHING_HEAD_ON = 4
    TAILGATING = 5
    NEAR_DOORING = 6
    DODGING_OBSTACLE = 7
    OTHER = 8

def _map_enum_value(value: Any, enum_class: Enum) -> str:
    """Safely attempts to map a numeric value to its corresponding Enum string."""
    if pd.isna(value):
        return "N/A"
    try:
        return enum_class(int(value)).name.lower()
    except (ValueError, KeyError):
        return "other"

# --- Parsing Logic ---

def parse_simra_text_file(filepath: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    """Reads the raw text file and splits it into Ride, Incident, and Metadata sections."""
    with open(filepath, encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]
        
    if not lines:
        raise ValueError(f"File is empty: {filepath}")

    # Locate the split point between incidents and continuous ride data
    separator_idx = next((i for i, line in enumerate(lines) if line.startswith(SEPARATOR_PREFIX)), None)
    
    if separator_idx is None:
        raise ValueError(f"No valid separator found in {filepath}")

    # 1. Extract Metadata
    metadata = {
        "source_file": filepath.name,
        "ride_id": filepath.stem,
        "file_version_raw": lines[0],
        "ride_section_version_raw": lines[separator_idx + 1]
    }

    # 2. Extract Data Sections
    incident_lines = lines[1:separator_idx]
    ride_lines = lines[separator_idx + 2:]

    incident_rows = list(csv.reader(incident_lines))
    ride_rows = list(csv.reader(ride_lines))

    incidents_df = pd.DataFrame(incident_rows[1:], columns=incident_rows[0])
    ride_df = pd.DataFrame(ride_rows[1:], columns=ride_rows[0])

    return ride_df, incidents_df, metadata

# --- Cleaning Logic ---

def clean_incident_data(incidents_df: pd.DataFrame, metadata: Dict[str, str], ride_start_timestamp: float) -> pd.DataFrame:
    """Standardizes types and maps Enums for the incident dataset."""
    df = incidents_df.copy().replace(r"^\s*$", pd.NA, regex=True)
    
    numeric_columns = [
        "key", "lat", "lon", "ts", "bike", "childCheckBox", "trailerCheckBox",
        "pLoc", "incident", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9", "scary", "i10"
    ]
    
    # Cast expected numeric columns to floats/ints safely
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Map enumerations to readable strings
    if "bike" in df.columns:
        df["bike_type"] = df["bike"].apply(_map_enum_value, enum_class=BikeType)
    if "pLoc" in df.columns:
        df["phone_location"] = df["pLoc"].apply(_map_enum_value, enum_class=PhoneLocation)
    if "incident" in df.columns:
        df["incident_type"] = df["incident"].apply(_map_enum_value, enum_class=IncidentType)

    # Calculate time relative to the start of the ride
    if "ts" in df.columns and pd.notna(ride_start_timestamp):
        df["incident_time_s_from_start"] = (df["ts"] - ride_start_timestamp) / 1000.0
    else:
        df["incident_time_s_from_start"] = pd.NA

    # Append metadata
    for key, value in metadata.items():
        df[key] = value

    return df

def clean_ride_data(ride_df: pd.DataFrame, metadata: Dict[str, str]) -> pd.DataFrame:
    """Standardizes types, interpolates missing values, and aligns timestamps for the telemetry dataset."""
    df = ride_df.copy().replace(r"^\s*$", pd.NA, regex=True)
    
    numeric_columns = [
        "lat", "lon", "X", "Y", "Z", "timeStamp", "acc", "a", "b", "c",
        "obsDistanceLeft1", "obsDistanceLeft2", "obsDistanceRight1", "obsDistanceRight2",
        "obsClosePassEvent", "XL", "YL", "ZL", "RX", "RY", "RZ", "RC"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if not df.empty and "timeStamp" in df.columns:
        df = df.sort_values("timeStamp").reset_index(drop=True)
        
        # Interpolate small gaps in sensor data (linear)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_area='inside')
        
        # Establish zero-indexed time
        ride_start_timestamp = df["timeStamp"].min()
        df["time_s_from_start"] = (df["timeStamp"] - ride_start_timestamp) / 1000.0
    else:
        df["time_s_from_start"] = pd.NA

    for key, value in metadata.items():
        df[key] = value

    return df

def build_imu_subset(ride_df: pd.DataFrame) -> pd.DataFrame:
    """Extracts only the columns required for the kinematic/IMU analysis phase."""
    required_columns = [
        "timeStamp", "time_s_from_start", "XL", "YL", "ZL", "a", "b", "c",
        "RX", "RY", "RZ", "RC", "source_file", "ride_id", "file_version_raw", "ride_section_version_raw"
    ]
    available_columns = [col for col in required_columns if col in ride_df.columns]
    return ride_df[available_columns].copy()

# --- Main Execution Function ---

def process_simra_file(raw_filepath: str, output_directory: str) -> Dict[str, Any]:
    """Orchestrates the parsing, cleaning, and saving of a single SimRa file."""
    start_time = time.perf_counter()
    raw_path = Path(raw_filepath)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    input_size_mb = raw_path.stat().st_size / (1024 * 1024)
    
    # 1. Parse
    raw_ride_df, raw_incidents_df, metadata = parse_simra_text_file(raw_path)
    
    # 2. Clean
    clean_ride_df = clean_ride_data(raw_ride_df, metadata)
    
    ride_start_ts = clean_ride_df["timeStamp"].min() if not clean_ride_df.empty else pd.NA
    clean_incidents_df = clean_incident_data(raw_incidents_df, metadata, ride_start_ts)
    
    imu_analysis_df = build_imu_subset(clean_ride_df)
    
    # 3. Save outputs
    ride_id = metadata["ride_id"]
    incidents_path = out_dir / f"{ride_id}_incidents_clean.parquet"
    ride_full_path = out_dir / f"{ride_id}_ride_clean_full.parquet"
    imu_path = out_dir / f"{ride_id}_ride_imu_analysis.parquet"
    
    clean_incidents_df.to_parquet(incidents_path, index=False)
    clean_ride_df.to_parquet(ride_full_path, index=False)
    imu_analysis_df.to_parquet(imu_path, index=False)
    
    output_size_mb = sum([p.stat().st_size for p in [incidents_path, ride_full_path, imu_path]]) / (1024 * 1024)
    execution_time = time.perf_counter() - start_time
    
    # 4. Return standard metrics for the orchestrator
    return {
        "ride_id": ride_id,
        "ride_row_count": len(clean_ride_df),
        "incident_row_count": len(clean_incidents_df),
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "execution_time_s": round(execution_time, 3)
    }