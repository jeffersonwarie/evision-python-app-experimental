import requests
from datetime import datetime
import os
from zipfile import ZipFile
from constants import (
    TRENDS_KEYWORDS,
    STATE_CODE_MAPPER,
)
from pytrends.request import TrendReq
import pandas as pd
import os
import time
from random import randint
from random import uniform
from typing import Optional
import backoff

DOWNLOAD_DIR = '/home/epidjhmw/public_html/pydata'

current_year = int(datetime.now().strftime("%Y"))
year_2022_id = 62  # the ID for year 2022 is 62 on CDC
end_id = current_year - 2022 + year_2022_id
start_id = end_id - 7

def ensure_download_dir():
    """Create download directory if it doesn't exist"""
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

def cdc_ilinet_downloader():
    """
    Downloads ilinet data for states and national level.
    Specify the absolute dir where you want to download the files.
    """
    ensure_download_dir()
    print(
        f"Beginning downloading CDC ILINet data for National Level. Following are the start and end SeasonsDT params {start_id}-{end_id}"
    )

    request_dict_states = {
        "AppVersion": "Public",
        "DatasourceDT": [{"ID": 0, "Name": "WHO_NREVSS"}, {"ID": 1, "Name": "ILINet"}],
        "RegionTypeId": 5,
        "SubRegionsDT": [
            {"ID": 1, "Name": "1"},
            {"ID": 2, "Name": "2"},
            {"ID": 3, "Name": "3"},
            {"ID": 4, "Name": "4"},
            {"ID": 5, "Name": "5"},
            {"ID": 6, "Name": "6"},
            {"ID": 7, "Name": "7"},
            {"ID": 8, "Name": "8"},
            {"ID": 9, "Name": "9"},
            {"ID": 10, "Name": "10"},
            {"ID": 11, "Name": "11"},
            {"ID": 12, "Name": "12"},
            {"ID": 13, "Name": "13"},
            {"ID": 14, "Name": "14"},
            {"ID": 15, "Name": "15"},
            {"ID": 16, "Name": "16"},
            {"ID": 17, "Name": "17"},
            {"ID": 18, "Name": "18"},
            {"ID": 19, "Name": "19"},
            {"ID": 20, "Name": "20"},
            {"ID": 21, "Name": "21"},
            {"ID": 22, "Name": "22"},
            {"ID": 23, "Name": "23"},
            {"ID": 24, "Name": "24"},
            {"ID": 25, "Name": "25"},
            {"ID": 26, "Name": "26"},
            {"ID": 27, "Name": "27"},
            {"ID": 28, "Name": "28"},
            {"ID": 29, "Name": "29"},
            {"ID": 30, "Name": "30"},
            {"ID": 31, "Name": "31"},
            {"ID": 32, "Name": "32"},
            {"ID": 33, "Name": "33"},
            {"ID": 34, "Name": "34"},
            {"ID": 35, "Name": "35"},
            {"ID": 36, "Name": "36"},
            {"ID": 37, "Name": "37"},
            {"ID": 38, "Name": "38"},
            {"ID": 39, "Name": "39"},
            {"ID": 40, "Name": "40"},
            {"ID": 41, "Name": "41"},
            {"ID": 42, "Name": "42"},
            {"ID": 43, "Name": "43"},
            {"ID": 44, "Name": "44"},
            {"ID": 45, "Name": "45"},
            {"ID": 46, "Name": "46"},
            {"ID": 47, "Name": "47"},
            {"ID": 48, "Name": "48"},
            {"ID": 49, "Name": "49"},
            {"ID": 50, "Name": "50"},
            {"ID": 51, "Name": "51"},
            {"ID": 52, "Name": "52"},
            {"ID": 54, "Name": "54"},
            {"ID": 55, "Name": "55"},
            {"ID": 56, "Name": "56"},
            {"ID": 58, "Name": "58"},
            {"ID": 59, "Name": "59"},
        ],
        "SeasonsDT": [{"ID": i, "Name": str(i)} for i in range(start_id, end_id)],
    }

    request_dict_national = {
        "AppVersion": "Public",
        "DatasourceDT": [{"ID": 0, "Name": "WHO_NREVSS"}, {"ID": 1, "Name": "ILINet"}],
        "RegionTypeId": 3,
        "SeasonsDT": [{"ID": i, "Name": str(i)} for i in range(start_id, end_id)],
    }

    url = "https://gis.cdc.gov/grasp/flu2/PostPhase02DataDownload"

    # ---------- National level --------------------
    print(f"[INFO]: Beginning downloading ILINET National data.")
    try:
        resp = requests.post(url, json=request_dict_national, allow_redirects=True)
    except Exception as ex:
        print("Failed to download national data")
        raise ex

    download_file_path = os.path.join(DOWNLOAD_DIR, "ILINET-National.zip")
    print(
        f"[INFO]: Finished downloading ILINET National data. Extracting it in {download_file_path}"
    )
    # writing zip
    with open(download_file_path, "wb") as f:
        f.write(resp.content)

    # extracting only ILINet.csv in the DOWNLOAD_DIR
    with ZipFile(download_file_path, "r") as zip_ref:
        zip_ref.extract("ILINet.csv", DOWNLOAD_DIR)

    # renaming the CSV
    os.rename(
        os.path.join(DOWNLOAD_DIR, "ILINet.csv"),
        os.path.join(DOWNLOAD_DIR, "ILINet-National.csv"),
    )
    os.remove(download_file_path)

    # ---------- State level --------------------
    print(f"[INFO]: Beginning downloading ILINET State data.")
    try:
        resp = requests.post(url, json=request_dict_states, allow_redirects=True)
    except Exception as ex:
        print("Failed to download state data")
        raise ex

    download_file_path = os.path.join(DOWNLOAD_DIR, "ILINET-State.zip")
    print(
        f"[INFO]: Finished downloading ILINET State data. Extracting it in {download_file_path}"
    )
    # writing zip
    with open(download_file_path, "wb") as f:
        f.write(resp.content)

    # extracting only ILINet.csv in the DOWNLOAD_DIR
    with ZipFile(download_file_path, "r") as zip_ref:
        zip_ref.extract("ILINet.csv", DOWNLOAD_DIR)

    # renaming the CSV
    os.rename(
        os.path.join(DOWNLOAD_DIR, "ILINet.csv"),
        os.path.join(DOWNLOAD_DIR, "ILINet-State.csv"),
    )
    os.remove(download_file_path)

def backoff_hdlr(details):
    """Handler for backoff decorator to log retries"""
    with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
        f.write(f"Backing off {details['wait']:0.1f} seconds after {details['tries']} tries\n")

@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=300,
    on_backoff=backoff_hdlr
)
def fetch_trends_with_retry(pytrends: TrendReq, term: str, timeframe: str, geo: str) -> Optional[pd.DataFrame]:
    """Fetch Google Trends data with retry logic"""
    try:
        # Random delay between 2 and 5 seconds
        time.sleep(uniform(2, 5))
        pytrends.build_payload(kw_list=[term], timeframe=timeframe, geo=geo)
        # Additional delay after building payload
        time.sleep(uniform(1, 3))
        df = pytrends.interest_over_time()
        if df is None or df.empty:
            raise Exception("Empty response from Google Trends")
            
        # Handle the pandas warning by explicitly managing data types
        df = df.reset_index().drop(columns="isPartial")
        df = df.infer_objects()  # Properly infer data types
        return df
    except Exception as e:
        with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
            f.write(f"Error fetching trends for {term} in {geo}: {str(e)}\n")
        raise

def trends_scraper() -> None:
    """Scrapes Google Trends data with improved rate limiting handling"""
    terms = TRENDS_KEYWORDS
    timeframe = "today 5-y"
    pytrends = TrendReq(hl="en-US", tz=360)  # Remove retry parameters as they're causing issues

    # First scrape national data
    with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
        f.write("Beginning national trends scraping\n")

    national_df = None
    for term in terms:
        try:
            column_df = fetch_trends_with_retry(pytrends, term, timeframe, "US")
            
            if national_df is None:
                national_df = column_df.copy(deep=True)
            else:
                national_df = national_df.merge(column_df, on="date")
            
            with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
                f.write(f"Successfully fetched national data for {term}\n")
            
        except Exception as e:
            with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
                f.write(f"Failed to fetch national data for {term}: {str(e)}\n")
            continue

    if national_df is not None and not national_df.empty:
        national_df["week_number"] = national_df["date"].dt.strftime("%U").astype(int)
        national_df["year"] = national_df["date"].dt.strftime("%Y").astype(int)
        
        output_file = os.path.join(DOWNLOAD_DIR, "google_trends-National.csv")
        national_df.to_csv(output_file, index=False)
        with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
            f.write(f"National data successfully saved to {output_file}. Shape: {national_df.shape}\n")
    else:
        with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
            f.write("No national data was collected successfully\n")

    # Now scrape state-level data
    complete_df = None
    for state_name, state_code in STATE_CODE_MAPPER.items():
        with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
            f.write(f"Processing state: {state_name} ({state_code})\n")

        state_df = None
        for term in terms:
            try:
                column_df = fetch_trends_with_retry(pytrends, term, timeframe, state_code)
                
                if state_df is None:
                    state_df = column_df.copy(deep=True)
                else:
                    state_df = state_df.merge(column_df, on="date")
                
                with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
                    f.write(f"Successfully fetched data for {term} in {state_name}\n")
                
            except Exception as e:
                with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
                    f.write(f"Failed to fetch {term} for {state_name}: {str(e)}\n")
                continue

        if state_df is not None:
            state_df["state"] = state_name
            state_df["state_code"] = state_code
            if complete_df is None:
                complete_df = state_df.copy(deep=True)
            else:
                complete_df = pd.concat([complete_df, state_df])

    if complete_df is not None and not complete_df.empty:
        complete_df["week_number"] = complete_df["date"].dt.strftime("%U").astype(int)
        complete_df["year"] = complete_df["date"].dt.strftime("%Y").astype(int)
        
        output_file = os.path.join(DOWNLOAD_DIR, "google_trends-State.csv")
        complete_df.to_csv(output_file, index=False)
        with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
            f.write(f"State data successfully saved to {output_file}. Shape: {complete_df.shape}\n")
    else:
        with open(os.path.join(DOWNLOAD_DIR, 'scraper_log.txt'), 'a') as f:
            f.write("No state data was collected successfully\n")

def scrape_cdc_trends_data():
    ensure_download_dir()
    cdc_ilinet_downloader()
    trends_scraper()

if __name__ == "__main__":
    # logging for cron debugging
    log_file = os.path.join(DOWNLOAD_DIR, 'scraper_log.txt')
    try:
        with open(log_file, 'a') as f:
            f.write(f"\n=== Script started at {datetime.now()} ===\n")
        scrape_cdc_trends_data()
        with open(log_file, 'a') as f:
            f.write(f"Script completed successfully at {datetime.now()}\n")
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Error occurred at {datetime.now()}: {str(e)}\n")
        raise
