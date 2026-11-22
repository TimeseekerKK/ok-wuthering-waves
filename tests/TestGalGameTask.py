import unittest
from unittest.mock import Mock

from ok.feature.Box import Box

from src.task.GalGameTask import GalGameTask


class TestGalGameTask(unittest.TestCase):
    def setUp(self):
        self.task = GalGameTask.__new__(GalGameTask)
        self.task._executor = Mock(method=Mock(width=1600, height=900))
        self.task.click = Mock()
        self.task.log_info = Mock()
        self.task._autosave_grades = Mock()
        self.task._read = Mock()
        self.task._active_date = None
        self.task.wait_until = lambda check, **_: check()
        self.lobby = [Box(1410, 470, 136, 34, name="档案读取"),
                      Box(1330, 590, 160, 40, name="开始新学期ノ"),
                      Box(1230, 750, 160, 40, name="继续旅程")]
        self.archive = [Box(130, 136, 160, 40, name="档案读取"),
                        Box(644, 194, 140, 32, name="自动存档"),
                        Box(194, 199, 90, 28, name="2周目")]

    def test_grade_threshold_is_per_attribute(self):
        for grades, expected in (("CCDDD", True), ("BADDD", True), ("DSSSS", False),
                                 ("SDSSS", False), ("DDDDD", False), (None, False)):
            with self.subTest(grades=grades):
                self.assertEqual(expected, self.task._meets_autosave_minimum(grades))

    def test_archive_decides_resume_or_new_semester(self):
        for grades, journey, resume in (("CCDDD", 2, True), ("DSSSS", 1, False)):
            with self.subTest(grades=grades):
                self.task._read.side_effect = [self.archive, self.lobby]
                self.task._autosave_grades.return_value = grades
                self.task.click.reset_mock()

                self.task._start_from_archive(self.lobby)

                self.assertEqual(journey, self.task._journey_number)
                self.assertEqual(not resume, self.task._starting_semester)
                self.assertEqual(3, self.task.click.call_count)
                self.task.click.assert_called_with(self.lobby[2 if resume else 1], after_sleep=2)

    def test_unreadable_grades_do_not_authorize_starting_a_new_save(self):
        self.task._read.return_value = self.archive
        self.task._autosave_grades.return_value = None

        with self.assertRaisesRegex(RuntimeError, "无法确认自动存档的评分"):
            self.task._start_from_archive(self.lobby)

        self.task.click.assert_called_once()

    def test_empty_autosave_ignores_manual_archive_grades(self):
        empty_archive = [self.archive[0], Box(400, 251, 147, 44, name="暂无存档")]
        self.task._read.side_effect = [empty_archive, self.lobby]

        self.assertEqual((self.lobby, False), self.task._choose_journey(self.lobby))
        self.task._autosave_grades.assert_not_called()

    def test_schedule_date_accepts_truncated_weekday(self):
        self.assertEqual((4, 22), self.task._date([Box(128, 48, 100, 27, name="04/22周")]))

    def test_handbook_uses_title_when_the_settings_icon_is_misread(self):
        title = Box(720, 160, 160, 28, name="学院手册")
        settings = Box(1290, 48, 95, 26, name="à设置")
        exit_button = Box(963, 777, 100, 28, name="退出游戏")

        self.assertIs(exit_button, self.task._settings([title, settings, exit_button]))

    def test_full_screen_choice_recognizes_wide_option(self):
        locked = Box(1020, 435, 217, 27, name="还是尽快准备备案吧。")
        available = Box(907, 589, 430, 28, name="如果可信，我当然也想选择相信小克先生")
        key = Box(792, 590, 20, 24, name="F")

        self.assertIs(available, self.task._full_choice([locked, available, key]))
