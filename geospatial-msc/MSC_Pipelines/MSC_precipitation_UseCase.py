# Databricks notebook source
# MAGIC %md
# MAGIC <h2>Anticipated Profits Using Precipitation Data</h2>
# MAGIC <p href=https://eccc-msc.github.io/open-data/usage/use-case_arthur/use-case_arthur_en/>Reference Code</p>

# COMMAND ----------

# install additional required libraries
%pip install tabulate
%pip install OWSLib

# COMMAND ----------

# import libraries and initiate parameters
from datetime import datetime, timedelta
import re
import warnings

# The following modules must first be installed to use 
# this code out of Jupyter Notebook
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy
from owslib.wms import WebMapService
import pandas
from tabulate import tabulate

# Ignore warnings from the OWSLib module
warnings.filterwarnings('ignore', module='owslib', category=UserWarning)

# Parameters choice
# Layer:
layer = 'REPS.DIAG.3_PRMM.ERGE5'
# Coordinates:
y, x = 49.288, -123.116
# Local time zone (in this exemple, the local time zone is UTC-07:00):
time_zone = -7

# COMMAND ----------

min_x, min_y, max_x, max_y = x - 0.25, y - 0.25, x + 0.25, y + 0.25
# WMS service connection
wms = WebMapService('https://geo.weather.gc.ca/geomet?SERVICE=WMS' +
                    '&REQUEST=GetCapabilities',
                    version='1.3.0',
                    timeout=300)
# Extraction of temporal information from metadata
def time_parameters(layer):
    start_time, end_time, interval = (wms[layer]
                                      .dimensions['time']['values'][0]
                                      .split('/')
                                      )
    iso_format = '%Y-%m-%dT%H:%M:%SZ'
    start_time = datetime.strptime(start_time, iso_format)
    end_time = datetime.strptime(end_time, iso_format)
    interval = int(re.sub(r'\D', '', interval))
    return start_time, end_time, interval


start_time, end_time, interval = time_parameters(layer)

# To use specific starting and ending time, remove the #
# from the next lines and replace the start_time and
# end_time with the desired values:
# start_time = 'YYYY-MM-DDThh:00'
# end_time = 'YYYY-MM-DDThh:00'
# fmt = '%Y-%m-%dT%H:%M'
# start_time = datetime.strptime(start_time, fmt) - timedelta(hours=time_zone)
# end_time = datetime.strptime(end_time, fmt) - timedelta(hours=time_zone)

# Calculation of date and time for available predictions
# (the time variable represents time at UTC±00:00)
time = [start_time]
local_time = [start_time + timedelta(hours=time_zone)]
while time[-1] < end_time:
    time.append(time[-1] + timedelta(hours=interval))
    local_time.append(time[-1] + timedelta(hours=time_zone))

# COMMAND ----------

# Loop to carry out the requests and extract the probabilities
def request(layer): 
    info = []
    pixel_value = []
    for timestep in time:
        # WMS GetFeatureInfo query
        info.append(wms.getfeatureinfo(layers=[layer],
                                       srs='EPSG:4326',
                                       bbox=(min_x, min_y, max_x, max_y),
                                       size=(100, 100),
                                       format='image/jpeg',
                                       query_layers=[layer],
                                       info_format='text/plain',
                                       xy=(50, 50),
                                       feature_count=1,
                                       time=str(timestep.isoformat()) + 'Z'
                                       ))
        # Probability extraction from the request's results
        text = info[-1].read().decode('utf-8')
        print(text)
        pixel_value.append(str(re.findall(r'value_0\s+\d*.*\d+', text)))
        pixel_value[-1] = float(
            re.sub('value_0 = \'', '', pixel_value[-1])
            .strip('[""]')
        )
    
    return pixel_value

pixel_value = request(layer)

# COMMAND ----------

# Function to adjust the alignment of two y axis
def align_yaxis(ax, ax2):
    y_lims = numpy.array([ax.get_ylim() for ax in [ax, ax2]])

    # Normalize both y axis
    y_magnitudes = (y_lims[:,1] - y_lims[:,0]).reshape(len(y_lims),1)
    y_lims_normalized = y_lims / y_magnitudes

    # Find combined range
    y_new_lims_normalized = numpy.array([numpy.min(y_lims_normalized), 
                                         numpy.max(y_lims_normalized)])

    # Denormalize combined range to get new axis
    new_lim1, new_lim2 = y_new_lims_normalized * y_magnitudes
    ax2.set_ylim(new_lim2)

    
# Function to create the plot
def fig(x, y, title, xlabel, ylabel, ylim, color = 'black', y2 = '', y2label = ''):
    # Plot and text size parameters
    params = {'legend.fontsize': '14',
              'figure.figsize': (8, 6),
              'axes.labelsize': '14',
              'axes.titlesize': '16',
              'xtick.labelsize': '12',
              'ytick.labelsize': '12'}
    plt.rcParams.update(params)

    # Plot creation and plot styling
    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o')

    # Titles
    plt.title(title)
    plt.xlabel(xlabel)
    ax.set_ylabel(ylabel, color=color)

    # Y axis range
    ax.set_ylim(ylim)
    ax.tick_params(axis='y', labelcolor=color)
    
    # Grid
    plt.grid(True, which='both')
    
    # Add a second dataset
    if y2 is not None:
        ax2 = plt.twinx()
        ax2.plot(x, y2, marker='o', color='tab:red')
        # Second y axis title
        ax2.set_ylabel(y2label, color='tab:red')
        # Range and ticks of second y axis
        ax2.set_ylim(0, (max(y2) * 1.1))
        ax2.tick_params(axis='y', labelcolor='tab:red')
        align_yaxis(ax, ax2)

    # Date format on x axis
    plt.gcf().autofmt_xdate()
    my_format = mdates.DateFormatter('%m/%d %H:%M')
    plt.gca().xaxis.set_major_formatter(my_format)

    # Graduation of x axis depending on the number of values plotted
    # Variable containing the hours for which there will be ticks: 
    hour = []
    for timestep in x:
        hour.append(int(timestep.strftime('%#H')))

    # Frequency of ticks and labels on the x axis
    if len(hour) < 8:
        # More precise graduation if there is only a few values plotted
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=hour))
    elif len(hour) > 8 and len(hour) <25:
        ax.xaxis.set_major_locator(mdates.HourLocator(byhour=hour, interval=2))
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=hour))
    else:
        # Coarser graduation if there is a lot of values plotted
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(0, 12)))
    
    return fig

# Add quantity of precipitations to the plot
# Verification of temporal parameters compatibility:
start_time1, end_time1, interval1 = time_parameters('REPS.DIAG.3_PRMM.ERMEAN')

if (start_time1 == start_time
        and end_time1 == end_time
        and interval1 == interval):
    # GetFeatureInfo request
    y2 = request(layer = 'REPS.DIAG.3_PRMM.ERMEAN')
    y2label = 'Quantity of precipitations (mm)'
else:
    y2 = None
    y2label = None

# Create the plot with the fig function and show the plot
fig(x = local_time,
    y = pixel_value,
    title = ('Probability of getting 5 mm or more of precipitations between' +
             f'\n{local_time[0]} and {local_time[-1]} (local time)'),
    xlabel = '\nDate and time',
    ylabel = 'Probability of getting 5 mm\nor more of precipitations (%)',
    ylim = (-10, 110),
    y2 = y2,
    y2label = y2label
    )

plt.show()

# COMMAND ----------

# Umbrellas sold per day in average when there is less than 30% 
# chance that there will be a minimum of 5 mm of precipitations
base = 3

# Profits per umbrella
umbrella_profit = 10.00

# Slope calculation data
# When the probability of precipitations is 30%...
x1 = 30
# ... 10 umbrellas are sold each hour
y1 = 10
# When the probability of precipitations is 100%...
x2 = 100
# ... 30 umbrellas are sold each hour
y2 = 30

# Slope calculation
slope = ((y2-y1)/(x2-x1))

# Open hours
opening = '09:00:00'
opening = datetime.strptime(opening, '%H:%M:%S')
closing = '21:00:00'
closing = datetime.strptime(closing, '%H:%M:%S')

# Prediction times that are within the open hours
open_hours = []
for timestep in local_time:
    if (timestep.time() > opening.time() 
            and timestep.time() < closing.time()):
        open_hours.append(timestep)

# Number of umbrellas sold each day independently of meteorological conditions
opening_interval = (opening + timedelta(hours=interval)).time()
umbrella = []
for timestep in local_time:
    new_day = timestep.time() < opening_interval
    if (umbrella == [] or new_day) and timestep in open_hours:
        umbrella.append(base)      
    else:
        umbrella.append(0)

# Number of umbrellas sold and anticipated profits
# depending on precipitations probability
cumulative_profit = []
for index, probability in enumerate(pixel_value):
    # Equation to calculate the number of umbrellas sold per hour
    eq = y1 + round((probability - 30) * slope)
    # Equation to calculate the number of umbrellas sold between 2 predictions
    eq2 = eq * interval 
    if local_time[index] in open_hours and probability > 30:
        if index == 0:
            umbrella[index] = umbrella[index] + eq2
        else: 
            umbrella[index] = umbrella[index] + umbrella[index - 1] + eq2
    elif index != 0:
        umbrella[index] = umbrella[index] + umbrella[index-1]                            
    cumulative_profit.append(umbrella[index] * umbrella_profit)



# Create and show plot
fig(x=local_time,
    y=pixel_value,
    title=('Anticipated profits from umbrellas sales ' +
             'depending\non precipitations probability'),
    xlabel='\nDate and time',
    ylabel='Probability of getting 5 mm\nor more of precipitations (%)',
    ylim=(-10, 110),
    y2=cumulative_profit,
    y2label='Cumulative anticipated profits ($)'
    )

plt.show()


# Probability of precipitations and cumulative
# profits only within open hours
probability = []
profit = []
for index, timestep in enumerate(local_time):
    if timestep in open_hours:
        probability.append(pixel_value[index])
        profit.append(cumulative_profit[index])

# Create table
columns = ['Local date and time', 'Probability', 'Anticipated cumulative profits ($)']
profit_df = spark.createDataFrame(zip(open_hours, probability, profit), columns)

# Show table
print('Anticipated profits from umbrellas sales ' +
      'depending on precipitations probability')
profit_df.show()

# Save in CSV format (remove # from the following lines)
# profit_df.to_csv('profit.csv',
#                 sep=';',
#                 index=False,
#                 encoding='utf-8-sig')
