import os
from wwo_hist import retrieve_hist_data


def get_weather_data(counties, start_date, end_date, frequency, output_dir):
    '''
    counties: (list) of counties, example: ['nyc', 'queens']
    start_date: (str) date, example: '01-JAN-2018'
    end_date: (str) date, example: '31-DEC-2018'
    frequency: (int) frequency of historical data, example: 1 for hourly data
    output_dir: (str) path to directory where you want to store the weather data
    '''
    os.chdir(output_dir)
    for county in counties:
        print(f"Getting weather data for {county}")
        api_key = '<your-api-key>'  # enter your api key for wwo
        location_list = [county]
        hist_weather_data = retrieve_hist_data(api_key,
                                        location_list,
                                        start_date,
                                        end_date,
                                        frequency,
                                        location_label = False,
                                        export_csv = True,
                                        store_df = True)
        
if __name__ == "__main__":
    counties = ['nyc']
    start_date = '01-JAN-2018'
    end_date = '31-DEC-2019'
    frequency = 1
    output_dir = "../data/weather_data"
    
    get_weather_data(counties, start_date, end_date, frequency, outpu_dir)

    