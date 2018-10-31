"""Unit tests for nowcast_update.py."""

# standard library
import argparse
import random
import unittest
from unittest.mock import MagicMock

# first party
from delphi.utils.epiweek import range_epiweeks

# py3tester coverage target
__test_target__ = 'delphi.nowcast_norovirus_private.nowcast_update'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def setUp(self):
    random.seed(12345)

  def test_get_argument_parser(self):
    """An ArgumentParser should be returned."""
    self.assertIsInstance(get_argument_parser(), argparse.ArgumentParser)

  def test_validate_args(self):
    """Arguments should be validated."""

    with self.subTest(name='first only'):
      args = MagicMock(first=2, last=None, test=False, target='tar')
      with self.assertRaises(Exception):
        validate_args(args)

    with self.subTest(name='last only'):
      args = MagicMock(first=None, last=1, test=False, target='tar')
      with self.assertRaises(Exception):
        validate_args(args)

    with self.subTest(name='first > last'):
      args = MagicMock(first=2, last=1, test=False, target='tar')
      with self.assertRaises(Exception):
        validate_args(args)

    with self.subTest(name='first < last'):
      args = MagicMock(first=1, last=2, test=False, target='tar')
      self.assertEqual(validate_args(args), (1, 2, False, 'tar'))

    with self.subTest(name='test mode'):
      args = MagicMock(first=None, last=None, test=True, target='tar')
      # self.assertEqual(validate_args(args), (None, None, True, target='tar'))
      with self.assertRaises(Exception):
        validate_args(args)

  def test_new_instance(self):
    """Create a NowcastUpdate instance with default parameters."""
    self.assertIsInstance(NowcastUpdate.new_instance(True, 'tar'), NowcastUpdate)

  def test_update(self):
    """Compute and store a nowcast."""

    database = MagicMock()
    database.__enter__.return_value = database
    database.__exit__.return_value = None
    data_source = MagicMock(
        get_truth_locations=lambda *a: ['pa', 'vi'],
        get_sensor_locations=lambda *a: ['pa', 'vi'],
        get_missing_locations=lambda *a: (),
        get_sensors=lambda *a: ['ght', 'sar3'],
        get_most_recent_issue=lambda *a: 201513,
        get_weeks=lambda *a: list(range_epiweeks(201413, 201514)),
        get_truth_value=lambda *a: random.random(),
        get_sensor_value=lambda *a: random.random(),
        prefetch=lambda *a: None)
    target = 'ov_noro_broad'

    NowcastUpdate(database, data_source, target).update(201512, 201513)

    self.assertEqual(database.set_last_update_time.call_count, 1)
    self.assertEqual(database.insert.call_count, 4)

    target_epiweek_location_triplets = set()
    for args, kwargs in database.insert.call_args_list:
      target_epiweek_location_triplets.add(args[:3])

    self.assertIn(('ov_noro_broad', 201512, 'pa'), target_epiweek_location_triplets)
    self.assertIn(('ov_noro_broad', 201513, 'pa'), target_epiweek_location_triplets)
    self.assertIn(('ov_noro_broad', 201512, 'vi'), target_epiweek_location_triplets)
    self.assertIn(('ov_noro_broad', 201513, 'vi'), target_epiweek_location_triplets)

  def test_get_update_range(self):
    """Get the range of epiweeks to be updated."""

    data_source = MagicMock(get_most_recent_issue=lambda *a: 201401)
    target = 'ov_noro_broad'
    updater = NowcastUpdate(None, data_source, target)

    first_week, last_week = updater.get_update_range(None, None)

    self.assertEqual(first_week, 201401)
    self.assertEqual(last_week, 201402)

    first_week, last_week = updater.get_update_range(200101, 200201)

    self.assertEqual(first_week, 200101)
    self.assertEqual(last_week, 200201)
