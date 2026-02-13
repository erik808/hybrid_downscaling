import numpy as np
import compute_tool


class Metrics():

    def __init__(self, dm):
        self.dm = dm
        self.ct = compute_tool.ComputeTool(dm=self.dm)
        self.metrics_dict = {}
        self.metrics_dict['RMSE'] = {}
        self.metrics_dict['correlation'] = {}
        self.trunc_time = 24*7

    def field_manip(self, data, field_type='all'):
        if field_type == 'uo':
            return data[..., 0]
        elif field_type == 'ssh':
            return data[..., 2]
        elif field_type == 'all':
            return data
        elif field_type == 'vorticity':
            return self.ct.vorticity(data, None)
        elif field_type == 'energy':
            return np.sum(np.square(data[..., :2]), axis=-1)
        else:
            raise Exception('invalid field_type')

    def compute_metric(self, data, metric='RMSE', field_type='all'):
        truth = self.field_manip(data['truth']['data'], field_type)

        for key, value in data.items():
            if key == 'truth':
                continue

            prediction = self.field_manip(value['data'], field_type)

            if metric == 'RMSE':
                self.compute_RMSE(truth, prediction, key, field_type)
            elif metric == 'correlation':
                self.compute_correlation(truth, prediction, key, field_type)

    def compute_RMSE(self, truth, prediction, key, field_type):
        shape = truth.shape
        # put spatial dim in vector form
        error = (truth - prediction).reshape(shape[0], -1)
        # truncate first week to get rid of startup effects
        error = error[self.trunc_time:,]
        # sum over space
        errnorm = np.sum(np.square(error), -1)
        # mean over time
        RMSE = np.sqrt(np.mean(errnorm))
        # add to dict
        if key in self.metrics_dict['RMSE']:
            self.metrics_dict['RMSE'][key].update({field_type: RMSE})
        else:
            self.metrics_dict['RMSE'][key] = {field_type: RMSE}

    def compute_correlation(self, truth, prediction, key, field_type):
        shape = truth.shape
        mid_lat = int(shape[1]/2)-1
        mid_lon = int(shape[2]/2)-1
        pred = prediction[self.trunc_time:, mid_lat, mid_lon]
        true = truth[self.trunc_time:, mid_lat, mid_lon]
        pred_std = np.std(pred)
        true_std = np.std(true)
        cov = np.cov(pred, true, rowvar=False)[1, 0]
        correlation = cov / (pred_std * true_std)

        if key in self.metrics_dict['correlation']:
            self.metrics_dict['correlation'][key]\
                .update({field_type: correlation})
        else:
            self.metrics_dict['correlation'][key] = \
                {field_type: correlation}
