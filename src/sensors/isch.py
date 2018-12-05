"""
For now, not going to worry about lags, don't have enough data
ISCH is a misnomer, SAR3 is a better definition, although not quite
Documentation from noro:
===============
=== Purpose ===
===============
Intercept-Sin-Cos-Holiday regression. It predicts wILI in some region on some
epiweek using ordinary regression. There are 6 covariates (7 if you count the
intercept term):
  - 4 indicator (0/1) variables for holiday weeks (50 or 51 through 01)
  - 2 timing variables: sin and cos of the epiweek
When producing retrospective predictions, great care is taken to only use
'valid' data: values that would have actually been available at the time.
However, unstable wILI is only available for recent years and for only some of
the regions (i.e. not in census regions). During training, ISCH will fall back
to stable data if unstable data is unavailable; however, during prediction,
ISCH will raise an Exception if unstable data is unavailable.
Note that the epiweek parameter represents the most recently published issue.
The returned value is a prediction for the following week.
See also:
  - arch.py: another system that generates 1-week-ahead predictions
=================
=== Changelog ===
=================
2016-04-11
  * allow predictions using invalid (stable) data
2016-04-06
  + initial version
"""

# standard library
import argparse

# third party
import numpy as np

# first party
from delphi.epidata.client.delphi_epidata import Epidata
import delphi.operations.secrets as secrets
import delphi.utils.epiweek as EW
from delphi.nowcast_dengue.util.dengue_data_source import DengueDataSource
from delphi.nowcast_dengue.util.cumulative_to_weekly import cum_to_week

def mutate_rows_as_if_lagged(rows, lag):
  for row in rows:
    row.update({'lag': lag})
  return rows


class ISCH:

  @staticmethod
  def dot(*Ms):
    N = Ms[0]
    for M in Ms[1:]:
      N = np.dot(N, M)
    return N

  def __init__(self, region, target, use_weekly = True):
    self.region = region
    self.target = target
    self.stts = 0
    weeks = Epidata.range(201401, 202330)
    rx = Epidata.check(Epidata.paho_dengue(self.region, weeks))
    self.data = {}
    self.valid = {}
    self.ew2i, self.i2ew = {}, {}
    for ew in EW.range_epiweeks(weeks['from'], weeks['to'], inclusive=True):
      # if 200916 <= ew <= 201015:
      #   continue
      i = len(self.ew2i)
      self.ew2i[ew] = i
      self.i2ew[i] = ew
    epiweeks = list(map(lambda elt: elt['epiweek'], rx))
    values = list(map(lambda elt: elt[self.target], rx))
    w_values = cum_to_week(epiweeks,values)
    for i in range(len(rx)):
      ew, observation = epiweeks[i], w_values[i]
      if ew not in self.ew2i:
        continue
      i = self.ew2i[ew]
      if i not in self.data:
        self.data[i] = {}
        self.valid[i] = {'stable': False}
      lag = 'stable'
      self.data[i][lag] = observation
      self.valid[i][lag] = True
    self.weeks = sorted(list(self.data.keys()))
    self.dds = DengueDataSource.new_instance(target)

  def _get_features(self, ew, signal_to_truth_shift=0, valid=False):
    X = np.zeros((1, 10))
    i = self.ew2i[ew]
    X[0, 0] = 1
    for lag in range(3):
        if valid and not self.valid[i-lag][lag]:
            w = self.i2ew[i-lag]
            raise Exception('missing unstable wILI (ew=%d|lag=%d)' % (w, lag))
        X[0, 1 + lag] = self.data[i-lag-signal_to_truth_shift]['stable']
    for holiday in range(4):
      if EW.split_epiweek(EW.add_epiweeks(ew, holiday))[1] == 1:
        X[0, 4 + holiday] = 1
    y, w = EW.split_epiweek(ew)
    N = EW.get_num_weeks(y)
    offset = np.pi * 2 * w / N
    X[0, 8] = np.sin(offset)
    X[0, 9] = np.cos(offset)
    # todo linear time trend covariate?
    return X

  def train(self, epiweek):
    if epiweek not in self.ew2i:
      raise Exception('not predicting during this period')
    most_recent_issue = self.dds.get_most_recent_issue(self.region)
    i2 = min(self.ew2i[epiweek] - 5, self.ew2i[most_recent_issue]-1)
    signal_to_truth_shift = EW.delta_epiweeks(most_recent_issue, epiweek)
    self.stts = signal_to_truth_shift
    i1 = self.weeks[2+signal_to_truth_shift]
    ew1, ew2 = self.i2ew[i1], self.i2ew[i2]
    num_weeks = i2 - i1
    X, Y = np.zeros((num_weeks, 10)), np.zeros((num_weeks, 1))
    r = 0
    for i in range(i1, i2):
      X[r, :] = self._get_features(self.i2ew[i], signal_to_truth_shift=signal_to_truth_shift, valid=False)
      Y[r, 0] = self.data[i + 1]['stable']
      r += 1
    self.model = ISCH.dot(np.linalg.inv(ISCH.dot(X.T, X)), X.T, Y)
    self.training_week = epiweek
    return (X, Y, self.model)

  def predict(self, epiweek, train=True, valid=False):
    if train:
      self.train(epiweek)
    if self.training_week > epiweek:
      raise Exception('trained on future data')
    X = self._get_features(epiweek, signal_to_truth_shift=self.stts, valid=valid)
    return float(ISCH.dot(X, self.model)[0, 0])


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('epiweek', type=int, help='most recently published epiweek (best 201030+)')
  parser.add_argument('region', type=str, help='region (state)')
  parser.add_argument('target', type=str, help='target (e.g., ov_noro_broad)')
  args = parser.parse_args()

  # options
  ew1, reg, tar = args.epiweek, args.region, args.target
  ew2 = EW.add_epiweeks(ew1, 1)

  # train and predict
  print('Most recent issue: %d' % ew1)
  prediction = ISCH(reg, tar).predict(ew1, True)
  print('Predicted observation for %s in %s on %d: %.3f' % (tar, reg, ew2, prediction))
  auth = invisible_secrets.invisible_secrets.optum_agg
  res = EpidataPrivate.optum_agg(auth, reg, ew2)
  if res['result'] == 1:
    row = res['epidata'][0]
    # issue = row['issue']
    observation = row[tar]
    err = prediction - observation
    print('Actual observation as of %s: %.3f (err=%+.3f)' % ('static report', observation, err))
  else:
    print('Actual observation: unknown')

# fixme may want to be forecasting proportions or rates
# todo may want Loch Ness intercept sensor instead or in addition to this one
