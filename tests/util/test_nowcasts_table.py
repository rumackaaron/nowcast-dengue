"""Unit tests for nowcasts_table.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# py3tester coverage target
__test_target__ = 'delphi.nowcast_norovirus_private.util.nowcasts_table'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_insert(self):
    """Insert a nowcast."""

    database = MagicMock()
    target, epiweek, location, value, stdev = 'tar', 1, 'a', 2, 3
    NowcastsTable(database=database).insert(target, epiweek, location, value, stdev)

    self.assertEqual(database.execute.call_count, 1)
    args, kwargs = database.execute.call_args
    sql, args = args
    self.assertEqual(sql, NowcastsTable.SQL_INSERT)
    self.assertEqual(args, (target, epiweek, location, value, stdev, value, stdev))

  def test_set_last_update_time(self):
    """Set the timestamp of last nowcast update."""

    database = MagicMock()
    NowcastsTable(database=database).set_last_update_time()

    self.assertEqual(database.execute.call_count, 1)
    args, kwargs = database.execute.call_args
    sql, args = args
    self.assertEqual(sql, NowcastsTable.SQL_INSERT)
    self.assertEqual(args[0], 'ov_noro_broad')
    self.assertEqual(args[1], 0)
    self.assertEqual(args[2], 'updated')
    self.assertIsInstance(args[3], int)
    self.assertIsInstance(args[4], int)
    self.assertEqual(args[3], args[5])
    self.assertEqual(args[4], args[6])

  def test_get_connection_info(self):
    """Return connection info."""

    username, password, database = NowcastsTable()._get_connection_info()
    self.assertIsInstance(username, str)
    self.assertIsInstance(password, str)
    self.assertIsInstance(database, str)
