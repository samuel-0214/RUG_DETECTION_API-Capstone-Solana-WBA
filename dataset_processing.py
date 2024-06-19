import requests
import dotenv
import time
import joblib
import os
import pandas as pd
import numpy as np
import json


dotenv.load_dotenv()
vybe_api = os.environ.get("VYBE_API")

# print(vybe_api) to check if the environment is set perfectly

#-------------------------------------------------------------------------------------

# Extracting the price history peaks

def get_token_price_history_with_retry(time_start,time_end, token_id,max_retries=3):
    url = f"https://api.vybenetwork.xyz/price/{token_id}/token-quote-ohlcv"

    headers = {
        "Content-Type": "application/json",
        'X-API-KEY': vybe_api
    }

    params = {
        "stride" : "1 hour",
        "time_end" : time_end,
        "time_start" : time_start
    }

    backingoff_time = 1 #initially is given as 1 in seconds

    for eattempt in range(max_retries):
        try:
            response = requests.get(url,headers=headers,params=params)

            if response.status_code==200 or response.staus_code==204:
                return response.json()
            elif response.status_code==409:
                print(f"Received 429 - Too Many Requests. Retrying in {backingoff_time} seconds for {token_id}.")
                time.sleep(backingoff_time)
                backingoff_time *= 2 #We can adjust this according to our needs
            else:
                print(f"Error :{response.status_code}-{response.text} for {token_id}. Retrying...")
                time.sleep(backingoff_time)
                backingoff_time *= 2 #We can adjust this according to our needs
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e},{token_id}")
            return(f"Request exception : {e}")
        
        backingoff_time *= 2

        if  eattempt < max_retries-1 :
            time.sleep(2)
    print(f"Maximum retries of {max_retries} has reached for {token_id}.")
    return None

#----------------------------------------------------------------------------------------

# Calculating Volatility using standard deviation

def calculate_volatility(result):
    Default_Volatility_Score = 0 #initialized with none 

    if 'data' in result:
        token_data = result['data']

        #Extracting relevant data and create a dataframe
        columns = ['timeBucketStart','open','high','low','close','count']
        data = pd.DataFrame(token_data, columns=columns)

        #converting timeBucketStart to datetime and set it to index
        data['timeBucketStart'] = pd.to_datetime(data['timeBucketStart'],unit='s')
        data = data.set_index('timeBucketStart')

        #converting numerical data to floating points
        numeric_columns = ['open','high','low','close']
        data[numeric_columns] = data[numeric_columns].astype(float)

        #calculating daily returns
        data['Daily_Returns'] = data['close'].pct_change()

        #calculating volatility
        volatility = np.std(data['Daily_Returns'])

        min_vol = np.min(data['Daily_Returns'])
        max_vol = np.max(data['Daily_Returns'])

        if np.isclose(max_vol, min_vol):
            print("Denominator is close to zero. Setting volatility score to default value.")
            return Default_Volatility_Score
        else:
            # Perform the division only if the denominator is not close to zero
            volatility_score = ((volatility - min_vol) / (max_vol - min_vol)) * 100

        return volatility_score
    else:
        return Default_Volatility_Score
    
#----------------------------------------------------------------------------------------

# Calculating Volatility 24 hour Change percentage

def cal_vola24hrChangePercentage(token_data):
    if 'data' in token_data and len(token_data['data']) >= 2:
        first_close = float(token_data['data'][0]['close'])
        last_close = float(token_data['data'][-1]['close'])
        vola24hrChangePercentage = ((last_close - first_close) / first_close) * 100
        return vola24hrChangePercentage
    return None

#----------------------------------------------------------------------------------------

#Calculating token details

def get_token_details(time_start,time_end, token_id,max_retries=3):
    url = f"https://api.vybenetwork.xyz/token/{token_id}"

    headers = {
        "Content-Type": "application/json",
        'X-API-KEY': vybe_api
    }

    backingoff_time = 1 #initially is given as 1 in seconds

    for eattempt in range(max_retries):
        try:
            response = requests.get(url,headers=headers)

            if response.status_code==200 or response.staus_code==204:
                return response.json()
            elif response.status_code==409:
                print(f"Received 429 - Too Many Requests. Retrying in {backingoff_time} seconds for {token_id}.")
                time.sleep(backingoff_time)
                backingoff_time *= 2 #We can adjust this according to our needs
            else:
                print(f"Error :{response.status_code}-{response.text} for {token_id}. Retrying...")
                time.sleep(backingoff_time)
                backingoff_time *= 2 #We can adjust this according to our needs
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {e},{token_id}")
            return(f"Request exception : {e}")
        
        backingoff_time *= 2

        if  eattempt < max_retries-1 :
            time.sleep(2)

    print(f"Maximum retries of {max_retries} has reached for {token_id}.")
    return None

#------------------------------------------------------------------------------------------------------

#Calculating liquidity 

def calculate_liquidity(token_data):
    if 'marketcap' in token_data and 'tokenAmountValue' in token_data:
        market_cap = token_data['marketcap']
        token_Amount = token_data['tokenAmountValue']
        if token_Amount is not None and token_Amount > 0:
            liquidity = market_cap / token_Amount 
            return liquidity
    return 0 

#------------------------------------------------------------------------------------------------------

#Calculating mysterious holders count 

def get_no_of_holder_count_mysterious(token_id, interval='day'):
    
    url = f"https://api.vybenetwork.xyz/tokens/{token_id}/holders-ts"

    header={
        "Content_Type": "application/json",
        "X-API-KEY": vybe_api
    }

    params={
        "interval": interval,
        "time_end":"null",
        "time_start":"null"
    }

    response = requests.get(url,headers=header,params=params)
    if response.status_code==200:
        data = response.json()
        return data['data'][-1]["nholders"]
    else:
        print(f"Failed to fetch the data: {response}")
        return None

#-----------------------------------------------------------------------------------------------------

#Making them into a structured json file 

async def fetchDataFunc(token_id):
    try:
        token_data = get_token_details(token_id)
        if token_data is None:
            print(f"Failed to fetch token details for token ID: {token_id}")
            return None
        
        time_start = int(time.time()) - (24 * 60 * 60)
        time_end = int(time.time())
        token_OHLCV_data = get_token_price_history_with_retry(time_start, time_end, token_id)
        if token_OHLCV_data is None:
            print(f"Failed to fetch token OHLCV data for token ID: {token_id}")
            return None
        
        v24hChangePercent = cal_vola24hrChangePercentage(token_OHLCV_data)
        liquidity = calculate_liquidity(token_data)
        volatility_score = calculate_volatility(token_OHLCV_data)
        holder_count = get_no_of_holder_count_mysterious(token_id)
        
        v24hUSD = 0
        if 'usdValueVolume' in token_data and token_data['usdValueVolume'] is not None:
            v24hUSD = token_data['usdValueVolume']
        
        input_data = {
            "decimals": token_data.get('decimal', 0),  # Default to 0 if 'decimal' key is missing
            "liquidity": liquidity,
            "logoURI": 1,   # Placeholder values
            "name": 1,      # Placeholder values
            "symbol": 1,    # Placeholder values
            "v24hChangePercent": v24hChangePercent,
            "v24hUSD": v24hUSD,
            "Volatility": volatility_score,
            "holders_count": holder_count
        }
        
        output_file = "processed_data.json"
        try:
            with open(output_file, 'w') as json_file:
                json.dump(input_data, json_file, indent=4)
            print(f"JSON file '{output_file}' created successfully for token ID: {token_id}")
            return input_data
        except Exception as e:
            print(f"Error occurred while creating JSON file for token ID {token_id}: {e}")
            return None
    
    except Exception as ex:
        print(f"Exception occurred while processing token ID {token_id}: {ex}")
        return None
