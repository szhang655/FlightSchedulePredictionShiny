
import ipyleaflet as L
from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import joblib
import pickle
from datetime import datetime, timedelta
import airportsdata
from faicons import icon_svg
from shinywidgets import output_widget, render_widget
from geopy.distance import  great_circle  
import pytz
from timezonefinder import TimezoneFinder
import requests
from math import atan
import traceback



# AIRPORT
airports = airportsdata.load('IATA')
us_airports = {code: data for code, data in airports.items() if data['country'] == 'US'}
us_iata_codes = list(us_airports.keys())

# MODEL
cancel_preprocessor = joblib.load('preprocessor_cancelled.pkl')
with open('xgb_model_cancelled.pkl', 'rb') as f:
    cancel_model = pickle.load(f)


delay_preprocessor = joblib.load('preprocessor_delayed.pkl')
with open('xgb_model_delayed.pkl', 'rb') as f:
    delay_model = pickle.load(f)


# API
API_KEY = 'c53937a42bddfc777aea71f5f9ec06ea'
UNITS = 'imperial'
EXCLUDE = 'minutely,hourly,alerts'

# 预测函数
def predict_cancel(flight_data):
    X_cancel = cancel_preprocessor.transform(pd.DataFrame([flight_data]))
    cancel = cancel_model.predict(X_cancel)
    return cancel[0]

def predict_delay(flight_data):
    X_delay = delay_preprocessor.transform(pd.DataFrame([flight_data]))
    delay_time = delay_model.predict(X_delay)
    return delay_time[0]

# 输出时间转换函数
def calculate_local_time(scheduled_time, delay_minutes, flight_date):
    delay_minutes = int(delay_minutes)  
    scheduled_datetime = datetime.combine(flight_date, datetime.strptime(scheduled_time, "%H:%M").time())
    arrival_datetime = scheduled_datetime + timedelta(minutes=delay_minutes)
    return arrival_datetime.strftime("%Y-%m-%d %H:%M")


# 获取机场数据，包括经纬度
def get_airport_data(airport_code):
    airport = us_airports.get(airport_code, None)
    if airport:
        return airport["lat"], airport["lon"]
    return None, None

# 计算两地之间的Great Circle Distance
def calculate_great_circle_distance(lat1, lon1, lat2, lon2):
    return  great_circle((lat1, lon1), (lat2, lon2)).miles  # 返回距离（英里）

# 动态计算假日天数的函数
def calculate_days_to_holidays(flight_date):
    # 获取航班的年份和月份
    year = flight_date.year
    month = flight_date.month

    # 设置假日日期，确保这些日期为 `date` 类型
    thanksgiving_date = datetime(year - 1 if month == 1 else year, 11, 24).date()
    christmas_date = datetime(year - 1 if month == 1 else year, 12, 25).date()
    new_year_date = datetime(year + 1 if month in [11, 12] else year, 1, 1).date()

    # 计算到每个假日的天数
    days_after_thanksgiving = (flight_date - thanksgiving_date).days
    days_after_christmas = (flight_date - christmas_date).days
    days_after_new_year = (flight_date - new_year_date).days

    return days_after_thanksgiving, days_after_christmas, days_after_new_year


# 输入验证函数
def validate_time_format(time_str):
    try:
        datetime.strptime(time_str, "%H:%M")
        return True
    except ValueError:
        return False

def validate_hour_minute(time_str):
    if validate_time_format(time_str):
        hour, minute = map(int, time_str.split(":"))
        if hour < 24 and minute < 60:
            return True
    return False

def validate_flight_number(flight_number):
    return flight_number[:2].isalpha() and len(flight_number) >= 2

def validate_date(date_str):
    return date_str.month in [11, 12, 1]  # 仅允许11月、12月和1月的日期

# Function to fetch weather data from the OpenWeather API
def get_weather_data(lat, lon):
    url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&exclude={EXCLUDE}&units={UNITS}&appid={API_KEY}"
    response = requests.get(url)
    return response.json()

# Function to get the weather data for the origin and destination airports
def get_airport_weather(flight_date, origin_lat, origin_lon, dest_lat, dest_lon):
    # Check if the flight date is valid (within 7 days in the future)
    today = datetime.today().date()
    flight_date = flight_date
    delta_days = (flight_date - today).days
    
    if delta_days < 0 or delta_days > 7:
        return None, "Weather data only available for dates within next 7 days"
    
    origin_weather = get_weather_data(origin_lat, origin_lon)
    destination_weather = get_weather_data(dest_lat, dest_lon)

    origin_day = origin_weather['daily'][delta_days]
    dest_day = destination_weather['daily'][delta_days]


    def process_weather_data(day_data):
        temp = day_data['temp']['day']
        humidity = day_data['humidity']
        pressure = day_data['pressure']
        wind_speed = day_data['wind_speed']
        precipitation = day_data.get('rain', 0)  # Rain
        snow = day_data.get('snow', 0)  # Snow

        wet_bulb_temp = temp * (
            atan(0.151977 * (humidity + 8.313659) ** 0.5) + atan(temp + humidity)
            - atan(humidity - 1.676331) + 0.00391838 * humidity ** 1.5 * atan(0.023101 * humidity) - 4.686035
        )

        return {
            "temperature": temp,
            "humidity_category": categorize_humidity(humidity),
            "pressure_category": categorize_pressure(pressure),
            "wet_bulb_temp_category": categorize_wet_bulb_temperature(wet_bulb_temp),
            "wind_speed_category": categorize_wind_speed(wind_speed),
            "precipitation_category": categorize_precipitation(precipitation),
            "snow_depth_category": categorize_snow_depth(snow),
            "extreme_weather": is_extreme_weather(wind_speed, humidity, wet_bulb_temp, precipitation)
        }

    origin_weather_data = process_weather_data(origin_day)
    origin_weather_data_ret = {
        "temperature_ORIGIN": origin_weather_data['temperature'],
        "HumidityCategory_ORIGIN" : origin_weather_data['humidity_category'],
        "WetBulbTempCategory_ORIGIN": origin_weather_data['wet_bulb_temp_category'],
        "PrecipitationCategory_ORIGIN": origin_weather_data['precipitation_category'],
        "PressureCategory_ORIGIN": origin_weather_data['pressure_category'],
        "SnowDepthCategory_ORIGIN": origin_weather_data['snow_depth_category'],
        "WindSpeedCategory_ORIGIN": origin_weather_data['wind_speed_category'],
        "EXTREME_ORIGIN": origin_weather_data['extreme_weather']
    }
    destination_weather_data = process_weather_data(dest_day)
    destination_weather_data_ret = {
        "temperature_DEST": destination_weather_data['temperature'],
        "WetBulbTempCategory_DEST": destination_weather_data['wet_bulb_temp_category'],
        "EXTREME_DEST": destination_weather_data['extreme_weather'],
        "PrecipitationCategory_DEST": destination_weather_data['precipitation_category'],
        "SnowDepthCategory_DEST":destination_weather_data['snow_depth_category'],
        "HumidityCategory_DEST": destination_weather_data['humidity_category'],
        "PressureCategory_DEST":destination_weather_data['pressure_category'],
        "WindSpeedCategory_DEST": destination_weather_data['wind_speed_category']
    }
    
    return {
        "origin": origin_weather_data_ret,
        "destination": destination_weather_data_ret
    }, None

# Weather data processing functions
def categorize_humidity(value):
    if value < 30:
        return 0
    elif 30 <= value < 50:
        return 1
    elif 50 <= value < 70:
        return 2
    elif 70 <= value < 85:
        return 3
    elif 85 <= value < 95:
        return 4
    else:
        return 5

def categorize_pressure(value):
    if value > 1020:
        return 0
    elif 1010 <= value <= 1020:
        return 1
    elif 1000 <= value < 1010:
        return 2
    elif 990 <= value < 1000:
        return 3
    elif 980 <= value < 990:
        return 4
    else:
        return 5

def categorize_wet_bulb_temperature(value):
    if value < 10:
        return 0
    elif 10 <= value < 15:
        return 1
    elif 15 <= value < 20:
        return 2
    elif 20 <= value < 25:
        return 3
    elif 25 <= value < 30:
        return 4
    else:
        return 5

def categorize_wind_speed(value):
    if 0 <= value < 5:
        return 0
    elif 5 <= value < 15:
        return 1
    elif 15 <= value < 30:
        return 2
    elif 30 <= value < 50:
        return 3
    elif 50 <= value < 70:
        return 4
    else:
        return 5

def categorize_snow_depth(value):
    if value == 0:
        return 0
    elif 0 < value <= 5:
        return 1
    elif 5 < value <= 15:
        return 2
    elif 15 < value <= 30:
        return 3
    elif 30 < value <= 50:
        return 4
    else:
        return 5

def categorize_precipitation(value):
    if value == 0:
        return 0
    elif 0 < value <= 2:
        return 1
    elif 2 < value <= 10:
        return 2
    elif 10 < value <= 30:
        return 3
    elif 30 < value <= 50:
        return 4
    else:
        return 5

def is_extreme_weather(wind_speed, humidity, wet_bulb_temp, precipitation):
    return (
        (wind_speed > 13.9) or                           # Wind speed > 13.9 m/s
        (humidity > 90 and wet_bulb_temp > 30) or        # Humidity > 90% and Wet Bulb Temp > 30°C
        (wet_bulb_temp < -15) or                         # Wet Bulb Temp < -15°C
        (precipitation > 10)                             # Precipitation > 10 mm
    )


# UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize("departure_airport", "Departure Airport", choices=us_iata_codes, selected="MSN"),
        ui.input_selectize("arrival_airport", "Arrival Airport", choices=us_iata_codes, selected="LGA"),
        ui.input_date("fl_date", "Flight Date"),
        ui.input_text("dep_time", "Scheduled Departure Time", placeholder='HH:MM'),
        ui.input_text("arr_time", "Scheduled Arrival Time", placeholder='HH:MM'),
        ui.input_text("fl_number", "Flight Number", placeholder='example: DL5806'),
        ui.input_text("op_unique_carrier", "Operating Carrier Flight Number", placeholder='example: DL5806'),
        ui.input_task_button("predict", "Predict Arrival Time"),
        ui.output_text("warning_message") 
    ),
    ui.layout_column_wrap(
        ui.value_box("Great Circle Distance", ui.output_text("great_circle_dist"), theme="gradient-blue-indigo", showcase=icon_svg("globe")),
        ui.value_box("Estimated Arrival Time", ui.output_text("arrival_time_output"), theme="gradient-blue-indigo", showcase=icon_svg("plane")),
        ui.value_box("Destination Airport Weather", ui.output_text("weather_info"), theme="gradient-blue-indigo", showcase=icon_svg("cloud")),
        height="220px",
    ),
    ui.card(
        ui.card_header("Map"),
        output_widget("map"),
        full_screen=True,
    ),
    ui.card(
        ui.card_header("Traveling Tips & Contact Information (Expandable)"
        ),
        ui.p("Avoid flying on Mondays to minimize both cancellation and delay risk."),
        ui.p("In December, try to avoid flights operated by JetBlue (B6) due to higher delays."),
        ui.p("In January, consider avoiding United Airlines (UA) flights, as they tend to have higher cancellation and delay rates during this month."),
        ui.p("Try not to schedule flights 2 days before major holidays or within 5 days after, as these are peak travel times with higher chances of delays and cancellations."),
        ui.p(""), 
        # ui.p("For Airlines: Allocate extra resources and improve scheduling during high cancellation times (mid-week, certain months, and holiday seasons"),  
        ui.p("** If you have any questions about the app, feel free to contact us at: hyunseung@stat.wisc.edu, szhang655@wisc.edu, clou25@wisc.edu, rming@wisc.edu, myu259@wisc.edu"),
        ui.card_footer("Copyright @ Fall2024-Stat628-Module3-Group5"),
        full_screen=True,
    ),
    title="Holiday Season(Nov-Jan) Flight Schedule Prediction",
    fillable=True,
    class_="bslib-page-dashboard",
)



# SERVER
def server(input, output, session):

    @reactive.event(input.predict)
    def check_inputs_and_predict():
        can_predict = True
        warning_message = ""

        if not input.departure_airport():
            can_predict = False
            warning_message += "Invalid departure Airport. .\n"
        
        if not input.arrival_airport():
            can_predict = False
            warning_message += "Invalid departure Airport. .\n"
        
        if not validate_hour_minute(input.dep_time()):
            can_predict = False
            warning_message += "Invalid departure time. Must be in HH:MM format with valid hour (0-23) and minute (0-59).\n"

        if not validate_hour_minute(input.arr_time()):
            can_predict = False
            warning_message += "Invalid arrival time. Must be in HH:MM format with valid hour (0-23) and minute (0-59).\n"

        if not validate_flight_number(input.fl_number()):
            can_predict = False
            warning_message += "Invalid flight number. Must start with two letters.\n"

        if not validate_flight_number(input.op_unique_carrier()):
            can_predict = False
            warning_message += "Invalid operating carrier flight number. Must start with two letters.\n"

        if not validate_date(input.fl_date()):
            can_predict = False
            warning_message += "Flight date must be in November, December, or January.\n"

        return warning_message, can_predict

    # 渲染警告信息
    @output
    @render.text
    def warning_message():
        message, can_predict = check_inputs_and_predict()
        return message if not can_predict else ""
    
    # 渲染到达时间
    @output
    @render.text
    def arrival_time_output():
        message, can_predict = check_inputs_and_predict()
        try:
            if can_predict:
                # 获取经纬度
                lat1, lon1 = get_airport_data(input.departure_airport())
                lat2, lon2 = get_airport_data(input.arrival_airport())

                # 计算地理距离（这部分已经在上面的 `great_circle_dist` 函数中做了处理）
                great_circle_distance = calculate_great_circle_distance(lat1, lon1, lat2, lon2)
                # 计算其他数据和预测结果
                flight_date = input.fl_date()

                # 计算距假日的天数
                days_after_thanksgiving, days_after_christmas, days_after_new_year = calculate_days_to_holidays(flight_date)
                
                # 计算起飞机场的cst时间
                tz_finder = TimezoneFinder()
                dep_timezone_str = tz_finder.timezone_at(lng=lon1, lat=lat1)
                dep_local_tz = pytz.timezone(dep_timezone_str)
                cst_tz = pytz.timezone('US/Central') 
                dep_scheduled_datetime = datetime.combine(flight_date, datetime.strptime(input.dep_time(), "%H:%M").time())
                dep_local_time = dep_local_tz.localize(dep_scheduled_datetime)  # 转为当地时区
                dep_cst_time = dep_local_time.astimezone(cst_tz) 


                # 计算预计到达的cst时间
                arr_timezone_str = tz_finder.timezone_at(lng=lon2, lat=lat2)
                arr_local_tz = pytz.timezone(arr_timezone_str)
                arr_scheduled_datetime = datetime.combine(flight_date, datetime.strptime(input.arr_time(), "%H:%M").time())
                arr_local_time = arr_local_tz.localize(arr_scheduled_datetime)  # 转为当地时区
                arr_cst_time = arr_local_time.astimezone(cst_tz) 

                

                # 更新到预测数据中
                flight_data = {
                    "MONTH": flight_date.month,
                    "DAY_OF_MONTH": flight_date.day,
                    "DAY_OF_WEEK": flight_date.isoweekday(),
                    "MKT_CARRIER": input.fl_number()[:2].upper(),
                    "OP_UNIQUE_CARRIER": input.op_unique_carrier().upper(),
                    "ORIGIN": input.departure_airport(),
                    "DEST": input.arrival_airport(),
                    "DISTANCE": great_circle_distance,
                    "Days_after_Thanksgiving": days_after_thanksgiving,
                    "Days_after_Christmas": days_after_christmas,
                    "Days_after_New_Year": days_after_new_year,
                    "Timechange_CRS_DEP_Time": dep_cst_time,
                    "Timechange_CRS_ARR_TIME": arr_cst_time,
                    # 添加缺失的列，赋值为np.nan
                    "WetBulbTempCategory_DEST": np.nan,
                    "EXTREME_DEST": np.nan,
                    "ParsedSkyCondition_DEST": np.nan,
                    "PrecipitationCategory_DEST": np.nan,
                    "HumidityCategory_ORIGIN": np.nan,
                    "ParsedSkyCondition_ORIGIN": np.nan,
                    "SnowfallCategory_DEST": np.nan,
                    "SnowfallCategory_ORIGIN": np.nan,
                    "WetBulbTempCategory_ORIGIN": np.nan,
                    "SnowDepthCategory_DEST": np.nan,
                    "HumidityCategory_DEST": np.nan,
                    "PrecipitationCategory_ORIGIN": np.nan,
                    "PressureCategory_ORIGIN": np.nan,
                    "SnowDepthCategory_ORIGIN": np.nan,
                    "WindSpeedCategory_ORIGIN": np.nan,
                    "PressureCategory_DEST": np.nan,
                    "EXTREME_ORIGIN": np.nan,
                    "WindSpeedCategory_DEST": np.nan,
                }

                # if weather data available
                today = datetime.today().date()
                weather_flight_date = flight_date
                delta_days = (weather_flight_date - today).days
                
                if delta_days >= 0 and delta_days < 7:
                    weather_data, error_message = get_airport_weather(weather_flight_date, lat1, lon1, lat2, lon2)
                    if weather_data != None:
                        flight_data["WetBulbTempCategory_DEST"] = weather_data['destination']['WetBulbTempCategory_DEST']
                        flight_data["PrecipitationCategory_DEST"]= weather_data['destination']['PrecipitationCategory_DEST']
                        flight_data["HumidityCategory_ORIGIN"]= weather_data['origin']['HumidityCategory_ORIGIN']
                        flight_data["WetBulbTempCategory_ORIGIN"]= weather_data['origin']['WetBulbTempCategory_ORIGIN']
                        flight_data["SnowDepthCategory_DEST"]=weather_data['destination']['SnowDepthCategory_DEST']
                        flight_data["HumidityCategory_DEST"]= weather_data['destination']['HumidityCategory_DEST']
                        flight_data["PrecipitationCategory_ORIGIN"]= weather_data['origin']['PrecipitationCategory_ORIGIN']
                        flight_data["PressureCategory_ORIGIN"]= weather_data['origin']['PressureCategory_ORIGIN']
                        flight_data["SnowDepthCategory_ORIGIN"]= weather_data['origin']['SnowDepthCategory_ORIGIN']
                        flight_data["WindSpeedCategory_ORIGIN"]= weather_data['origin']['WindSpeedCategory_ORIGIN']
                        flight_data["PressureCategory_DEST"]= weather_data['destination']['PressureCategory_DEST']
                        flight_data["EXTREME_ORIGIN"]= weather_data['origin']['EXTREME_ORIGIN']
                        flight_data["WindSpeedCategory_DEST"]= weather_data['destination']['WindSpeedCategory_DEST']
                    

                        
    

                cancel = predict_cancel(flight_data)
                if cancel == 1:
                    message += "Flight may be canceled."
                    return 'Flight may be canceled.'
                else:
                    delay_minutes = predict_delay(flight_data)
                    arrival_time_local = calculate_local_time(input.arr_time(), delay_minutes, flight_date)
                    message += ", not likely to be canceled."
                    # output.arrival_time_output.set(arrival_time_local)
                    return arrival_time_local+message

            else:
                return message

        except Exception as e:
            message += f"Error occurred: {str(e)}"
            traceback.print_exc() 
            return message 
        
        

    # 渲染地理距离
    @output
    @render.text
    def great_circle_dist():

        message, can_predict = check_inputs_and_predict()
        try:
            if can_predict:
                # 获取经纬度
                lat1, lon1 = get_airport_data(input.departure_airport())
                lat2, lon2 = get_airport_data(input.arrival_airport())

                if lat1 is not None and lat2 is not None:
                    # 计算地理距离
                    great_circle_distance = calculate_great_circle_distance(lat1, lon1, lat2, lon2)
                    return f"{great_circle_distance:.2f} miles"
                else:
                    message += "Invalid airport codes"
                    return message
            else:
                return message
        except Exception as e:
            message += f"Error occurred: {str(e)}"
            return message
        
    @output
    @render.text
    def weather_info():
        message, can_predict = check_inputs_and_predict()
        try:
            if can_predict:
                lat1, lon1 = get_airport_data(input.departure_airport())
                lat2, lon2 = get_airport_data(input.arrival_airport())

                today = datetime.today().date()
                flight_date = input.fl_date()
                delta_days = (flight_date - today).days
                
                weather_data, error_message = get_airport_weather(flight_date, lat1, lon1, lat2, lon2)
                if weather_data == None:
                    return error_message
                else:
                    return str(weather_data['destination']['temperature_DEST']) + '°F'
            else:
                return message
        except Exception as e:
            message += f"Error occurred: {str(e)}"
            return message

    # # 预测航班状态
    # @reactive.effect
    # def make_prediction():
    #     pass
    
    def remove_layer(map: L.Map, name: str):
        for layer in map.layers:
            if layer.name == name:
                map.remove_layer(layer)

    def update_marker(map: L.Map, loc: tuple, name: str):
        remove_layer(map, name)
        m = L.Marker(location=loc, draggable=True, name=name)
        map.add_layer(m)

    def update_line(map: L.Map, loc1: tuple, loc2: tuple):
        remove_layer(map, "line")
        map.add_layer(
            L.Polyline(locations=[loc1, loc2], color="blue", weight=2, name="line")
        )



    @render_widget
    def map():
        return L.Map(zoom=6, center=(43.0731,-89.4012)) #default is madison
    
    @reactive.effect
    def _():
        message, can_predict = check_inputs_and_predict()
        try:
            if can_predict:
                lat1, lon1 = get_airport_data(input.departure_airport())
                lat2, lon2 = get_airport_data(input.arrival_airport())
                update_marker(map.widget,(lat1, lon1), "loc1")
                update_marker(map.widget,(lat2, lon2), "loc2")
                update_line(map.widget, (lat1, lon1),(lat2, lon2))

                lat_rng = [min(lat1,lat2), max(lat1,lat2)]
                lon_rng = [min(lon1,lon2), max(lon1,lon2)]
                new_bounds = [
                    [lat_rng[0], lon_rng[0]],
                    [lat_rng[1], lon_rng[1]],
                ]

                b = map.widget.bounds
                if len(b) == 0:
                    map.widget.fit_bounds(new_bounds)
                elif (
                    lat_rng[0] < b[0][0]
                    or lat_rng[1] > b[1][0]
                    or lon_rng[0] < b[0][1]
                    or lon_rng[1] > b[1][1]
                ):
                    map.widget.fit_bounds(new_bounds)
            else:
                return message
        except Exception as e:
            message += f"Error occurred: {str(e)}"
            return message
            

# 运行应用
app = App(app_ui, server)
