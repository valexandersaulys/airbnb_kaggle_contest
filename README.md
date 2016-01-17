# Introduction

## TO-DO
- Move all my mlp crap in ./prototype_alpha/ to its own folder (in general a good idea to start segregating)
- Try using committee-voting style (use Pandas to pick the most counted value for a given row, put into its own object?)
- Create totebag(!)

## General Direction
I decided to try something different for this and implement a type of templating
system. This way I can see changes as they occur and have perfect records of
them. However first I decided to build a script to import, join, and build the
requisite CSVs for predictions.

### Prototypes Alpha
Will look at just the train_users.csv features. No other features will be extrapolated
for predictions. Also will only predict one, will move on to using probabilities to guess
two different destinations later.

I am currently having issues as the neural networks I'm attempting to train get "stuck"
after only a round or two. My other methods (mostly from scikit) are ok, but fairly mediocre
in performance.


### Prototypes Beta
Dimensionality Reduction into two components. Try methods on this.

## CSV files and my observations

- train_users.csv & test_users.csv
All the relevant information. 
```
['id', 'date_account_created', 'timestamp_first_active', 'date_first_booking', 'gender', 'age', 'signup_method', 's
ignup_flow', 'language', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first
_device_type', 'first_browser', 'country_destination']
```
Shape of Test:  ( 43673,15)
Shape of Train: (171239,16)

- sessions.csv
Contains user_id for joining with the main user csv files.
Shape: (5600850,6)
```
['user_id', 'action', 'action_type', 'action_detail', 'device_type', 'secs_elapsed']
```

- countries.csv
Looks useless, like straight up. Contents coppied below
```
['country_destination', 'lat_destination', 'lng_destination', 'distance_km', 'destination_km2', 'destination_langua
ge ', 'language_levenshtein_distance']                                                                            
```

- age_gender_bkts.csv
Contains statistics on countries based on destination. Might be able to correlate
something like "People like to be with people their age so this might be a popular 
destination(?).
```
['age_bucket', 'country_destination', 'gender', 'population_in_thousands', 'year']
```


## Table of Files
- ./data_info.txt
Describes the csv files in question here.

- All *.csv files
Kept them here for the same of ease relating to processing. Its a pain to keep
them in their own folder and have to

- ./venv folder
Keeps all my relevant python packages here and such so I can revisit the work
if I want.

- ./build_join.py
Imports, builds, and joins all the raw data into a useful csv file.

- ./templates/
Holds all my template script files

- ./templates/