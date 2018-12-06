"""
(Nonmembers are not allowed to view this file.)

Ignoring issues for now, not enough data to use
Documentation from noro:
===============
=== Purpose ===
===============

Produces a signal for each flu digital surveillance source, which is then used
as a 'sensor' in the context of nowcasting through sensor fusion.

Each signal is updated over the following inclusive range of epiweeks:
  - epiweek of most recently computed signal of this type
  - last epiweek
The idea is to recompute the last stored value (just in case there were
changes to the underlying data source), and to compute all weeks up to, but
not including, the current week (because the current week is, by definition,
still ongoing).

The following signals are available:
  - gft: Google Flu Trends
  - ght: Google Health Trends
  - twtr: HealthTweets
  - wiki: Wikipedia access
  - cdc: CDC Page Hits
  - epic: Epicast 1-week-ahead point prediction
  - quid: Flu lab test data
  - sar3: Seasonal Autoregression (order 3)
  - arch: Best-fit Archetype at 1-week-ahead

See also:
  - signal_update.py
  - sar3.py
  - arch.py
"""
# standard library
import argparse
import re
import subprocess
import sys

# third party
import numpy as np

# first party
from delphi.epidata.client.delphi_epidata import Epidata
#from delphi.nowcast_dengue.sensors.arch import ARCH
#from delphi.nowcast_dengue.sensors.sar3 import SAR3
from delphi.nowcast_dengue.sensors.isch import ISCH
from delphi.nowcast_dengue.util.sensors_table import SensorsTable
from delphi.nowcast_dengue.util.cumulative_to_weekly import cum_to_week
import delphi.operations.secrets as secrets
from delphi.utils.epidate import EpiDate
import delphi.utils.epiweek as flu
from delphi.utils.geo.locations import Locations

 def get_most_recent_issue(epidata):
   # search for FluView issues within the last 10 weeks
   ew2 = EpiDate.today().get_ew()
   ew1 = flu.add_epiweeks(ew2, -9)
   rows = epidata.check(epidata.fluview('nat', epidata.range(ew1, ew2)))
   return max([row['issue'] for row in rows])

"""
Suggestions:
1. add paramters for functions (such as fit_loch_ness) which can specify "flu" or "norovirus" or other kinds of data
2. move functions such as dot to utils.py to make the code more concise
"""


class SignalGetter:
  """Class with static methods that implement the fetching of
  different data signals. Each function returns a function that
  only takes a single argument:
  - weeks: an Epiweek range of weeks to fetch data for.
  """
  def __init__(self):
    pass

  @staticmethod
  def get_ght(location, epiweek, valid):
    if location.upper() != 'US':
      location = location + "_"
    fetch = lambda weeks: Epidata.ght(secrets.api.ght, location, weeks, '/m/09wsg')
    return fetch


class SensorFitting:
  def __init__(self):
    pass

  @staticmethod
  def fit_loch_ness(location, epiweek, name, field, fetch, valid, target):
    # target_type is added for compatibility for other type of targets such as norovirus data

    # Helper functions
    def get_weeks(epiweek):
      ew1 = 201401
      ew2 = epiweek
      ew3 = flu.add_epiweeks(epiweek, 1)
      weeks0 = Epidata.range(ew1, ew2)
      weeks1 = Epidata.range(ew1, ew3)
      return (ew1, ew2, ew3, weeks0, weeks1)

    def extract(rows, field, to_weekly=False):
      data = {}
      for row in rows:
        data[row['epiweek']] = float(row[field])
      if not to_weekly:
        return data

      ews = list(data.keys())
      c_vals = [data[ew] for ew in ews]
      w_vals = cum_to_week(ews, c_vals)
      data = {}
      ews = sorted(ews)
      for i in range(len(ews)):
        data[ews[i]] = w_vals[i]
      return data

    def get_training_set_data(data):
      epiweeks = sorted(list(data.keys()))
      X = [data[ew]['x'] for ew in epiweeks]
      Y = [data[ew]['y'] for ew in epiweeks]
      return (epiweeks, X, Y)

    def get_training_set(location, epiweek, signal, valid):
      ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
      result = Epidata.paho_dengue(location, weeks0)
      rows = Epidata.check(result)
      stable = extract(rows, 'num_dengue', to_weekly=True)
      data = {}
      for ew in signal.keys():
        if ew == ew3 or ew not in stable:
          continue
        sig = signal[ew]
        num_dengue = stable[ew]
        data[ew] = {'x': sig, 'y': num_dengue}
      return get_training_set_data(data)

    def dot(*Ms):
      """ Simple function to compute the dot product
      for any number of arguments.
      """
      N = Ms[0]
      for M in Ms[1:]:
        N = np.dot(N, M)
      return N

    def get_weight(ew1, ew2):
      """ This function gives the weight between two given
      epiweeks based on a function that:
        - drops sharply over the most recent ~3 weeks
        - falls off exponentially with time
        - puts extra emphasis on the past weeks at the
          same time of year (seasonality)
        - gives no week a weight of zero
      """
      dw = flu.delta_epiweeks(ew1, ew2)
      yr = 52.2
      hl1, hl2, bw = yr, 1, 4
      a = 0.05
      # b = (np.cos(2 * np.pi * (dw / yr)) + 1) / 2
      b = np.exp(-((min(dw % yr, yr - dw % yr) / bw) ** 2))
      c = 2 ** -(dw / hl1)
      d = 1 - 2 ** -(dw / hl2)
      return (a + (1 - a) * b) * c * d

    def get_periodic_bias(epiweek):
      weeks_per_year = 52.2
      offset = flu.delta_epiweeks(201401, epiweek) % weeks_per_year
      angle = np.pi * 2 * offset / weeks_per_year
      return [np.sin(angle), np.cos(angle)]

    def apply_model(epiweek, beta, values):
      bias0 = [1.]
      if beta.shape[0] > len(values) + 1:
        # constant and periodic bias
        bias1 = get_periodic_bias(epiweek)
        obs = np.array([values + bias0 + bias1])
      else:
        # constant bias only
        obs = np.array([values + bias0])
      return float(dot(obs, beta))

    def get_model(ew2, epiweeks, X, Y):
      ne, nx1, ny = len(epiweeks), len(X), len(Y)
      if type(X[0]) == type([]):
        nx2 = len(X[0])
      else:
        nx2 = 1
      if ne != nx1 or nx1 != ny:
        raise Exception('length mismatch e=%d X=%d Y=%d' % (ne, nx1, ny))
      weights = np.diag([get_weight(ew1, ew2) for ew1 in epiweeks])
      X = np.array(X).reshape((nx1, nx2))
      Y = np.array(Y).reshape((ny, 1))
      bias0 = np.ones(Y.shape)
      if ne >= 26 and flu.delta_epiweeks(epiweeks[0], epiweeks[-1]) >= 52:
        # constant and periodic bias
        bias1 = np.array([get_periodic_bias(ew) for ew in epiweeks])
        X = np.hstack((X, bias0, bias1))
      else:
        # constant bias only
        X = np.hstack((X, bias0))
      XtXi = np.linalg.inv(dot(X.T, weights, X))
      XtY = dot(X.T, weights, Y)
      return np.dot(XtXi, XtY)

    ew1, ew2, ew3, weeks0, weeks1 = get_weeks(epiweek)
    rows = Epidata.check(fetch(weeks1))
    signal = extract(rows, field)
    min_rows = 4

    if ew3 not in signal:
      raise Exception('%s unavailable on %d' % (name, ew3))
    if len(signal) < min_rows:
      raise Exception('%s available less than %d weeks' % (name, min_rows))

    epiweeks, X, Y = get_training_set(location, epiweek, signal, target)

    min_rows = min_rows - 1
    if len(Y) < min_rows:
      raise Exception('paho_dengue available less than %d weeks' % (min_rows))

    model = get_model(ew3, epiweeks, X, Y)
    value = apply_model(ew3, model, [signal[ew3]])
    return value


class UnknownLocationException(Exception):
  """An Exception indicating that the given location is not known."""


def get_location_list(loc: str):
  locs = 'AG,AI,AW,BB,BS,CL,CU,GD,GF,GP,GY,HT,KN,LC,MQ,PA,SR,TC,TT,VC,AR,BM,BO,BR,BZ,CA,CO,CR,DM,DO,EC,GT,HN,JM,KY,MS,MX,NI,PE,PR,PY,SV,US,UY,VE'.split(',')
  """Return the list of locations described by the given string."""
  loc = loc.lower()
  if loc == 'all':
    return locs
  elif loc in locs:
    return [loc]
  else:
    raise UnknownLocationException('unknown location: %s' % str(loc))


class SensorGetter:
  """Class that implements different sensors. Some sensors
  may take in a signal to do the fitting on, others do not.
  """
  def __init__(self):
    pass
  
  @staticmethod
  def get_sensor_implementations():
    """Return a map from sensor names to sensor implementations."""
    return {
      'isch': SensorGetter.get_isch,
      'ght': SensorGetter.get_ght,
    }

  @staticmethod
  def get_isch(location, epiweek, valid, target):
    return ISCH(location, target).predict(epiweek, valid=valid)

  # sensors using the loch ness fitting

  @staticmethod
  def get_ght(location, epiweek, valid, target):
    fetch = SignalGetter.get_ght(location, epiweek, valid)
    return SensorFitting.fit_loch_ness(location, epiweek, 'ght', 'value', fetch, valid, target)


class SensorUpdate:
  """
  Produces both real-time and retrospective sensor readings for paho_dengue in the Americas.
  Readings (predictions of paho_dengue made using raw inputs) are stored in the Delphi
  database and are accessible via the Epidata API.
  """

  @staticmethod
  def new_instance(valid, test_mode, target):
    """
    Return a new instance under the default configuration.

    If `test_mode` is True, database changes will not be committed.

    If `valid` is True, be punctilious about hiding values that were not known
    at the time (e.g. run the model with preliminary observations only). Otherwise, be
    more lenient (e.g. fall back to final observations when preliminary data isn't
    available).
    """
    database = SensorsTable(test_mode=test_mode)
    implementations = SensorGetter.get_sensor_implementations()
    return SensorUpdate(valid, database, implementations, Epidata, target)

  def __init__(self, valid, database, implementations, epidata, target):
    self.valid = valid
    self.database = database
    self.implementations = implementations
    self.epidata = epidata
    self.target = target

  def update(self, sensors, first_week, last_week):
    """
    Compute sensor readings and store them in the database.
    """

    # most recent issue
    if last_week is None:
      last_issue = get_most_recent_issue(self.epidata)
      last_week = flu.add_epiweeks(last_issue, +1)

    # connect
    with self.database as database:

      # update each sensor
      for (name, loc) in sensors:

        # update each location
        for location in get_location_list(loc):

          # timing
          ew1 = first_week
          if ew1 is None:
            ew1 = database.get_most_recent_epiweek(name, location)
            if ew1 is None:
              # If an existing sensor reading wasn't found in the database and
              # no start week was given, just assume that readings should start
              # at 2014w01.
              ew1 = 201401
              print('%s-%s not found, starting at %d' % (name, location, ew1))

          args = (name, location, ew1, last_week)
          print('Updating %s-%s from %d to %d.' % args)
          for test_week in flu.range_epiweeks(ew1, last_week, inclusive=True):
            self.update_single(database, test_week, name, location)

  def update_single(self, database, test_week, name, location):
    train_week = flu.add_epiweeks(test_week, -1)
    impl = self.implementations[name]
    try:
      value = impl(location, train_week, self.valid, self.target)
      print(' %4s %5s %d -> %.3f' % (name, location, test_week, value))
    except Exception as ex:
      value = None
      print(' failed: %4s %5s %d' % (name, location, test_week), ex)
    if value is not None:
      database.insert(self.target, name, location, test_week, value)
    sys.stdout.flush()


def get_argument_parser():
  """Define command line arguments and usage."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'names',
      help=(
        'list of name-location pairs '
        '(location is two-letter country abbreviations)'))
  parser.add_argument(
      '--first',
      '-f',
      type=int,
      help='first epiweek override')
  parser.add_argument(
      '--last',
      '-l',
      type=int,
      help='last epiweek override')
  parser.add_argument(
      '--epiweek',
      '-w',
      type=int,
      help='epiweek override')
  parser.add_argument(
      '--target',
      type=str,
      default='num_dengue',
      help='The target column in dengue ground truth (PAHO) data')
  parser.add_argument(
      '--test',
      '-t',
      default=False,
      action='store_true',
      help='dry run only')
  parser.add_argument(
      '--valid',
      '-v',
      default=False,
      action='store_true',
      help='do not fall back to stable num_dengue; require unstable num_dengue')
  return parser


def validate_args(args):
  """Validate and return command line arguments."""

  # check epiweek specification
  first, last, week = args.first, args.last, args.epiweek
  for ew in [first, last, week]:
    if ew is not None:
      flu.check_epiweek(ew)
  if week is not None:
    if first is not None or last is not None:
      raise ValueError('`week` overrides `first` and `last`')
    first = last = week
  if first is not None and last is not None and first > last:
    raise ValueError('`first` must not be greater than `last`')

  # validate and extract name-location pairs
  pair_regex = '[^-,]+-[^-,]+'
  names_regex = '%s(,%s)*' % (pair_regex, pair_regex)
  if not re.match(names_regex, args.names):
    raise ValueError('invalid sensor specification')

  return args.names, first, last, args.valid, args.test, args.target


def parse_sensor_location_pairs(names):
  return [pair.split('-') for pair in names.split(',')]


def main(names, first, last, valid, test, target):
  """Run this script from the command line."""
  sensors = parse_sensor_location_pairs(names)
  SensorUpdate.new_instance(valid, test, target).update(sensors, first, last)


if __name__ == '__main__':
  main(*validate_args(get_argument_parser().parse_args()))
