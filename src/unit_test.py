import unittest
import json
import run_flask_server

# ------------------------------------------------------------------------------#
#                                    TESTS                                      #
# ------------------------------------------------------------------------------#


class TestServer(unittest.TestCase):
    def setUp(self):
        self.app = run_flask_server.app.test_client()

    def test_request_email_prediction(self):
        endpoint = '/predict'
        test_ts = [16768, 15735, 16048, 18946, 19536, 19594, 31500, 30761, 37395,
                   46011, 47722, 40644, 46352, 51401, 60968]
        min_val = 1000
        max_val = 300000
        post_response = self.app.post(endpoint, data=json.dumps(dict(ts_to_match=test_ts,
                                                                     min_val=min_val,
                                                                     max_val=max_val)),
                                      content_type='application/json')
        expected_response = {'prediction': 168522.92148886315,
                             'success': True}
        self.assertEqual(json.loads(post_response.data), expected_response)


if __name__ == '__main__':
    unittest.main()
