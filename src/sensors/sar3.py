"""
===============
=== Purpose ===
===============

Seasonal Autoregression of order 3. It predicts wILI in some region on some
epiweek using ordinary regression. There are 9 covariates (10 if you count the
intercept term):
  - 3 most recent (unstable) wILI values
  - 4 indicator (0/1) variables for holiday weeks (50 or 51 through 01)
  - 2 timing variables: sin and cos of the epiweek

When producing retrospective predictions, great care is taken to only use
'valid' data: values that would have actually been available at the time.
However, unstable wILI is only available for recent years and for only some of
the regions (i.e. not in census regions). During training, SAR3 will fall back
to stable data if unstable data is unavailable; however, during prediction,
SAR3 will raise an Exception if unstable data is unavailable.

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

def mutate_rows_as_if_lagged(rows, lag):
  for row in rows:
    row.update({'lag': lag})
  return rows


class SAR3:

  @staticmethod
  def dot(*Ms):
    N = Ms[0]
    for M in Ms[1:]:
      N = np.dot(N, M)
    return N

  def __init__(self, region, target):
    self.region = region
    self.target = target
    weeks = Epidata.range(201401, 202330)
    r0 = Epidata.check(Epidata.paho_dengue(self.region, weeks, lag=0))
    r1 = Epidata.check(Epidata.paho_dengue(self.region, weeks, lag=1))
    r2 = Epidata.check(Epidata.paho_dengue(self.region, weeks, lag=2))
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
    for row in r0 + r1 + r2 + rx:
      ew, observation, lag = row['epiweek'], row[self.target], row['lag']
      if ew not in self.ew2i:
        continue
      i = self.ew2i[ew]
      if i not in self.data:
        self.data[i] = {}
        self.valid[i] = {0: False, 1: False, 2: False, 'stable': False}
      if not (0 <= lag <= 2):
        lag = 'stable'
      self.data[i][lag] = observation
      self.valid[i][lag] = True
    self.weeks = sorted(list(self.data.keys()))
    for i in self.weeks:
      if 'stable' not in self.data[i]:
        continue
      for lag in range(3):
        if lag not in self.data[i]:
          self.data[i][lag] = self.data[i]['stable']

  def _get_features(self, ew, valid=True):
    X = np.zeros((1, 10))
    i = self.ew2i[ew]
    X[0, 0] = 1
    for lag in range(3):
      if valid and not self.valid[i - lag][lag]:
        w = self.i2ew[i - lag]
        raise Exception('missing unstable %s (ew=%d|lag=%d)' % (self.target, w, lag))
      X[0, 1 + lag] = self.data[i - lag][lag]
    for holiday in range(4):
      if EW.split_epiweek(EW.add_epiweeks(ew, holiday))[1] == 1:
        X[0, 4 + holiday] = 1
    y, w = EW.split_epiweek(ew)
    N = EW.get_num_weeks(y)
    offset = np.pi * 2 * w / N
    X[0, 8] = np.sin(offset)
    X[0, 9] = np.cos(offset)
    return X

  def train(self, epiweek):
    if epiweek not in self.ew2i:
      raise Exception('not predicting during this period')
    i1 = self.weeks[2]
    i2 = self.ew2i[epiweek] - 5
    ew1, ew2 = self.i2ew[i2], self.i2ew[i2]
    num_weeks = i2 - i1 + 1
    X, Y = np.zeros((num_weeks, 10)), np.zeros((num_weeks, 1))
    r = 0
    for i in range(i1, i2 + 1):
      X[r, :] = self._get_features(self.i2ew[i], valid=False)
      Y[r, 0] = self.data[i + 1]['stable']
      r += 1
    self.model = SAR3.dot(np.linalg.inv(SAR3.dot(X.T, X)), X.T, Y)
    self.training_week = epiweek
    return (X, Y, self.model)

  def predict(self, epiweek, train=True, valid=True):
    if train:
      self.train(epiweek)
    if self.training_week > epiweek:
      raise Exception('trained on future data')
    X = self._get_features(epiweek, valid=valid)
    return float(SAR3.dot(X, self.model)[0, 0])


if __name__ == '__main__':
  # args and usage
  parser = argparse.ArgumentParser()
  parser.add_argument('epiweek', type=int, help='most recently published epiweek')
  parser.add_argument('region', type=str, help='region (state)')
  parser.add_argument('target', type=str, help='target (e.g., num_dengue)')
  args = parser.parse_args()

  # options
  ew1, reg, tar = args.epiweek, args.region, args.target
  ew2 = EW.add_epiweeks(ew1, 1)

  # train and predict
  print('Most recent issue: %d' % ew1)
  prediction = SAR3(reg, tar).predict(ew1, True)
  print('Predicted observation for %s in %s on %d: %.3f' % (tar, reg, ew2, prediction))
  res = Epidata.paho_dengue(reg, ew2)
  if res['result'] == 1:
    row = res['epidata'][0]
    # issue = row['issue']
    observation = row[tar]
    err = prediction - observation
    print('Actual observation as of %s: %.3f (err=%+.3f)' % ('static report', observation, err))
  else:
    print('Actual observation: unknown')

  # fixme want to forecast deltas, not cumulative
