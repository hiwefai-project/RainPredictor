In this directory, the trained model's weights are.

To download pre-trained weights:

```bash
wget https://data.meteo.uniparthenope.it/instruments/rdr0/hiwefai_best_model.tar.gz --no-check-certificate

tar -xvzf hiwefai_best_model.tar.gz

rm hiwefai_best_model.tar.gz
bash```

Test the RainPrediction model with different trained modes.
Check which one works better with your data using utils/compare.py

**Important**: if you are using the Italian Department of Civil Protection weather radar, before training, set all no-data values to 0 and all values less than 0 to 0.
