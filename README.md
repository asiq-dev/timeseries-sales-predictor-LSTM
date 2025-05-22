# LSTM Stock Price Prediction

This project predicts Microsoft stock closing prices using an LSTM (Long Short-Term Memory) neural network built with TensorFlow/Keras.

## Features
- Loads historical Microsoft stock data from CSV
- Preprocesses and scales the data
- Trains an LSTM model to predict future closing prices
- Visualizes actual vs. predicted prices

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies

## Usage
1. Clone the repository or download the files.
2. (Optional) Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Make sure `MicrosoftStock.csv` is in the project directory.
5. Run the main script:
   ```powershell
   python main.py
   ```

## Output
- The script will display a plot comparing actual and predicted closing prices.

## Notes
- The model uses only the 'close' price for prediction.
- You can adjust the model architecture or parameters in `main.py`.

## License
This project is for educational purposes.
