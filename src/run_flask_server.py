"""run_flask_server.py                  Flask server for revenue forecasting"""
__author__ = "Edwin Zhang <edwin.james.zhang@gmail.com>"

# Necessary imports used in code
import logging.config
import json
import time
import flask
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Custom-written imports
from extract_candidates import CandidateExtractor
from log_utils import gen_error_msg
import traceback

# ------------------------------------------------------------------------------#
#                                 PARAMETERS                                    #
# ------------------------------------------------------------------------------#
timestr = time.strftime('%Y%b%d_%H%M%S%Z')  # timestamp string
COMP_DATA_PATH = '../data/Commerce analysis - full sales data.xlsx'
# ------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------#
#                    FLASK APP AND LOGGING INITIALIZATION                       #
# ------------------------------------------------------------------------------#
# Initialize our Flask application and the model
app = flask.Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='{"levelname": %(levelname)s, "name": %(name)s,'
                                               ' "message": %(message)s}')
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------#
comp_data = pd.read_excel(COMP_DATA_PATH, sheet_name = 'Sales')
comp_names = list(comp_data.Company.unique())
comp_data_transform = [comp_data.Sales[comp_data.Company == company].values for company in comp_names]

@app.route("/predict", methods=['POST'])
def predict():
    """ Given a time series, forecast one-year growth using similar time series

    Args (as POST request JSON input to endpoint):
        tsToMatch (list): contains the time series to forecast
        minVal (int): time series to compare must start and remain above this value, optional
        maxVal (int): time series to compare must not go above this value, optional

    Returns:
        flask.jsonify(data): JSON response containing model prediction and a success marker
    """
    # Initialize the data dictionary that will be returned from the view
    data = {'success': False}

    try:
        # Track prediction time
        start_time = time.time()

        # Read general inputs (TODO: add input validation)
        request_data = flask.request.get_json()
        ts_to_match = request_data.get('ts_to_match')
        min_val = request_data.get('min_val')
        max_val = request_data.get('max_val')
        
        # Log the request
        logger.info(json.dumps({'ts_to_match': ts_to_match, 'min_val': min_val,
                                'max_val': max_val}))
        
        if ts_to_match:
            ts_to_match = np.array(ts_to_match)
        
        # Get comp companies, filtering if specified
        candidate_extractor = CandidateExtractor(ts_to_match, comp_data_transform, comp_names)
        if min_val and max_val:
            candidate_extractor.filter_ts_candidates(min_val, max_val)
        best_candidates, best_distances, best_indices, forecast_vals, candidate_order = \
        candidate_extractor.get_best_candidates(5, norm = True, dtw = True)
        
        # Apply difference transform to ensure stationarity of time series
        ts_to_match_diff = ts_to_match[1:] - ts_to_match[:-1]
        candidate_ts_diff = [ts[1:] - ts[:-1] for ts in best_candidates]
        forecast_vals_diff = [forecast_val - ts[-1] for forecast_val, ts in zip(forecast_vals, 
                                                                                best_candidates)]
        # Create model and use it to forecast growth (TODO: serialize model object and save for re-use)
        model = LinearRegression()
        model.fit(X = np.array(candidate_ts_diff), y = forecast_vals_diff)
        data['prediction'] = model.predict(ts_to_match_diff.reshape(1, -1))[0]
        
        # Indicate that the request was a success
        data['success'] = True
        end_time = time.time()

        # Log the response
        logger.info(json.dumps({'data': data, 'prediction_time': end_time - start_time}))

    except Exception as ex:
        request_data = flask.request.get_json()
        print(traceback.format_exc())
        logger.error(json.dumps({'ts_to_match': request_data.get('ts_to_match'),
                                 'min_val': request_data.get('min_val'),
                                 'max_val': request_data.get('max_val'),
                                 'exception': gen_error_msg(ex)}))

    # Return the data dictionary as a JSON response
    return flask.jsonify(data)


# If this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    logger.info('Server started @ {0}'.format(timestr))
    app.run()
