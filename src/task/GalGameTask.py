import re
from collections import deque
from time import monotonic

import cv2
import numpy as np

from ok import BaseTask

from src.task.task_icons import task_icon


DATE = re.compile(r"(0[1-9]|1[0-2])[/／](0[1-9]|[12]\d|3[01])(?:周[一二三四五六日天]?)?")
REWARD_PAGES = ("限时奖励", "常驻奖励")
GIFT_LOCATIONS = ("街道", "学生理事会", "文科教室")
REWARD_PROGRESS = re.compile(r"(\d+)\s*[/／]\s*(14|15)")
LETTER_PAGE = re.compile(r"[1-9]/[1-9]")
AUTOSAVE_MINIMUM = "CCDDD"
GRADE_ORDER = "DCBAS"
# Colors of the five attribute grades on the archive cards, in BGR order.
ARCHIVE_GRADE_COLORS = {
    "D": (170, 172, 170),
    "C": (160, 185, 153),
    "B": (214, 172, 134),
    "A": (125, 102, 237),
    "S": (78, 173, 240),
}
STORY_CHOICES = (
    "之前在列车", "在列车上", "昨天在列车", "昨天在火车", "昨天在电车", "回到原本的世界", "我想回去",
    "是的",
    "休息一下再回去", "同时加入", "两者都加入", "两边都加入", "两个都加入",
    "我会利用自己的个人影响力", "在天马集团之前买下", "综合素质", "学业与逻辑", "洛可可帮忙",
    "学生理事会", "回家吧", "开启下一周目", "开启新周目",
)
COURSES = {
    "理科教室": ("课堂助教", "进阶理科课", "理科课程"),
    "文科教室": ("进阶文科课", "文科课程"),
    "操场": ("协助训练", "日常锻炼"),
    "家庭餐厅": ("前台收银", "点单服务"),
    "中央广场": ("校园清洁",),
    "学院商店": ("店员帮忙", "商品销售", "货物整理"),
    "外事联络中心": ("物资采购",),
    "学生理事会": ("学院宣传", "文书处理"),
    "社团联合": ("活动运营", "协助训练"),
    "海滨": ("滨海采风",),
}
OUTINGS = (
    "礼堂", "教学楼", "入学报到", "百年校庆", "期末综合测验", "期末测验", "预选赛", "暑期作战",
    "检定考试", "成人礼庆典", "成人礼", "全国大赛", "最终考察", "毕业典礼",
)
EXAM_ANSWERS = {
    (7, 1): (r"判断.*(?:感受|立场)", r"^斜边的平方$", r"(?:两者|二者|都).*(?:一样|相同|相等)"),
    (7, 2): (r"弦乐器", r"表达.*(?:情感|思想)", r"(?:舞台|表演).*(?:生动|统一|完整)"),
    (9, 2): (r"^4倍$", r"^25$", r"^28$"),
    (9, 3): (r"竖琴", r"明暗关系", r"低音提琴"),
    (11, 26): (r"^入射角$", r"一手.*二手|一次资料.*二次资料"),
    (11, 27): (r"音高.*响度|音高.*音量|声音的高低.*声音的大小", r"^打击乐(?:器)?$"),
}


class GalGameTask(BaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "3.8 GalGame"
        self.description = "从「第二索拉：未名之梦」活动页开始，推进学期并领取可用活动奖励。仅支持简体中文。"
        self.group_name = "Daily Activities"
        self.group_icon = task_icon("category-daily")
        self.supported_languages = ["zh_CN"]

    def _read(self):
        self.next_frame()
        return self.ocr(0, 0, 1, .987)

    def _label(self, boxes, label, region=(0, 0, 1, 1), *, contains=False):
        x1, y1, x2, y2 = region
        for box in boxes:
            x, y = box.center()
            text = re.sub(r"\s+", "", box.name)
            if (x1 <= x / self.width <= x2 and y1 <= y / self.height <= y2
                    and (label in text if contains else label == text)):
                return box
        return None

    def _lobby(self, boxes):
        return self._label(boxes, "开始新学期", (.65, .58, .98, .80), contains=True)

    def _archive_button(self, boxes):
        return self._label(boxes, "档案读取", (.84, .4, .99, .58), contains=True)

    def _archive_page(self, boxes):
        return self._label(boxes, "档案读取", (.07, .12, .27, .22), contains=True)

    def _autosave_grades(self):
        # Each attribute has its own slot. Read the main letter's color;
        # the superscript '+' does not affect comparison with C or D.
        grades = []
        y1, y2 = (round(value * self.height / 900) for value in (271, 299))
        for x in (356, 421, 486, 551, 616):
            x1, x2 = (round(value * self.width / 1600) for value in (x - 14, x + 14))
            pixels = self.frame[y1:y2, x1:x2].astype(np.int16)
            coverage = {
                grade: np.mean(np.all(np.abs(pixels - color) <= 14, axis=2))
                for grade, color in ARCHIVE_GRADE_COLORS.items()
            }
            grade = max(coverage, key=coverage.get)
            if coverage[grade] < .13:
                return None
            grades.append(grade)
        return "".join(grades)

    def _choose_journey(self, boxes):
        if self._archive_button(boxes):
            self.click(.922, .468, after_sleep=.6)
        elif not self._archive_page(boxes):
            raise RuntimeError("活动首页没有识别到「档案读取」。")

        grades = None
        journey_number = None

        def read_autosave():
            nonlocal grades, journey_number
            archive = self._read()
            if not self._archive_page(archive):
                return False
            if self._label(archive, "自动存档", (.40, .20, .50, .27), contains=True):
                grades = self._autosave_grades()
                if self._label(archive, "1周目", (.10, .18, .20, .27), contains=True):
                    journey_number = 1
                elif self._label(archive, "2周目", (.10, .18, .20, .27), contains=True):
                    journey_number = 2
                return grades is not None
            return bool(self._label(archive, "暂无存档", (.20, .24, .39, .36)))

        if not self.wait_until(read_autosave, time_out=6):
            raise RuntimeError("无法确认自动存档的评分，未选择旅程。")

        self.click(.954, .068, after_sleep=.65)

        def read_lobby():
            current = self._read()
            return current if self._lobby(current) and self._archive_button(current) else None

        lobby = self.wait_until(read_lobby, time_out=6)
        if not lobby:
            raise RuntimeError("检查自动存档后无法返回活动首页。")
        resume = self._meets_autosave_minimum(grades)
        self._journey_number = journey_number if resume else 1
        action = "继续旅程" if resume else "开始新学期"
        self.log_info(f"自动存档评分 {grades or '暂无存档'}，选择「{action}」")
        return lobby, resume

    @staticmethod
    def _meets_autosave_minimum(grades):
        return bool(grades and len(grades) == len(AUTOSAVE_MINIMUM)
                    and all(GRADE_ORDER.index(grade) >= GRADE_ORDER.index(minimum)
                            for grade, minimum in zip(grades, AUTOSAVE_MINIMUM)))

    def _continue_journey_button(self, boxes):
        button = self._label(boxes, "继续旅程", (.65, .72, .98, .96), contains=True)
        if button:
            return button
        return next(iter(self.ocr(.64, .72, .98, .96, match="继续旅程")), None)

    def _start_from_archive(self, boxes):
        lobby, resume = self._choose_journey(boxes)
        button = self._continue_journey_button(lobby) if resume else self._lobby(lobby)
        if not button:
            raise RuntimeError("自动存档评分已达标，未识别到「继续旅程」。")
        self._starting_semester = not resume
        self.click(button, after_sleep=2)

    def _date(self, boxes):
        for box in boxes:
            if box.x < .14 * self.width and box.y < .12 * self.height:
                if match := DATE.fullmatch(box.name.replace(" ", "")):
                    return tuple(map(int, match.groups()))
        return None

    def _resume_state(self, boxes):
        return (
            self._lobby(boxes) or self._settings(boxes) or self._date(boxes) or self._dialogue_panel(boxes)
            or self._confirm_event(boxes) or self._study_focus(boxes) or self._full_choice(boxes)
            or self._label(boxes, "是否快速跳过剧情", (.23, .3, .8, .58), contains=True)
            or self._label(boxes, "好感度提升", (.25, .24, .55, .36), contains=True)
            or self._label(boxes, "新增物品", (.06, .2, .33, .4), contains=True)
            or self._label(boxes, "属性提升", (.35, .26, .65, .39))
            or self._label(boxes, "本月总结", (.16, .03, .42, .14))
            or self._schedule_save(boxes) or self._ending_selector(boxes)
            or self._label(boxes, "即将进入结局阶段", (.29, .38, .73, .59), contains=True)
            or self._commission_offer(boxes)
            or self._attribute_check(boxes)
            or self._attribute_result(boxes)
            or self._lottery_screen(boxes)
            or self._shop_screen(boxes)
            or self._label(boxes, "「理科课程」", (.2, .34, .8, .59), contains=True)
            or self._forum(boxes)
            or self._paper_choice(boxes)
            or self._label(boxes, "课题清单", (0, 0, .2, .12), contains=True)
            or self._label(boxes, "档案管理", (0, 0, .2, .12), contains=True)
            or any(self._label(boxes, title, (0, 0, .2, .13)) for title in REWARD_PAGES)
            or self._letter_button(boxes)
            or self._letter(boxes)
            or self._envelope(boxes)
            or self._black_narration(boxes)
        )

    def _confirm_event(self, boxes):
        prompts = ("是否外出", "选择该方向后", "是否开始执行", "是否开始日程", "是否执行日程",
                   "将结束今日的地图探索")
        if self._starting_semester:
            if self._label(boxes, "覆盖当前自动存档", (.20, .38, .80, .57), contains=True):
                return self._label(boxes, "确认", (.56, .55, .78, .70))
            prompts += ("是否开始新学期", "是否开启新学期")
        if any(self._label(boxes, label, (.2, .34, .8, .59), contains=True) for label in prompts):
            return self._label(boxes, "确认", (.61, .75, .8, .9))
        return None

    def _study_focus(self, boxes):
        academic = self._label(boxes, "学业的追求与逻辑的锻炼", (.46, .28, .92, .44), contains=True)
        if not academic or self._journey_number == 1:
            return academic
        focus = self._label(boxes, "综合素质的全面成长与发展", (.46, .68, .92, .84), contains=True)
        locked = self._label(boxes, "后解锁", (.46, .81, .92, .95), contains=True)
        if focus and not locked:
            return focus
        return academic

    def _ending_selector(self, boxes):
        if not self._label(boxes, "选择属于你的", (.02, .14, .44, .28), contains=True):
            return None
        return self._label(boxes, "确定", (.40, .89, .60, .97))

    def _save_ending_checkpoint(self, confirm):
        self.click(confirm, after_sleep=.9)
        for _ in range(6):
            boxes = self._read()
            if storage := self._label(boxes, "档案存储", (.28, .65, .48, .93)):
                break
            if self._envelope(boxes):
                self.click(.50, .58, after_sleep=.65)
            elif self._letter_button(boxes):
                self.click(.895, .872, after_sleep=.65)
            elif self._dialogue_panel(boxes) or self._label(boxes, "长按加速", (.83, .86, 1, .97)):
                self.click(.90, .067, after_sleep=.65)
            elif self._date(boxes) == (12, 2):
                self.click(.954, .067, after_sleep=.65)
            else:
                self.sleep(.55)
        else:
            raise RuntimeError("结局前无法打开学院手册。")

        self.click(storage, after_sleep=.8)
        boxes = self._read()
        if not (self._label(boxes, "档案存储", (.07, .12, .29, .22), contains=True)
                and self._label(boxes, "12/02", (.10, .23, .21, .34))):
            raise RuntimeError("结局前未确认 12/02 存档页面。")
        empty_slots = [box for box in boxes if box.name == "暂无存档"
                       and .20 < box.center()[0] / self.width < .88
                       and .25 < box.center()[1] / self.height < .85]
        if not empty_slots:
            raise RuntimeError("结局前未找到空白手动存档位。")
        slot = min(empty_slots, key=lambda box: (box.center()[1], box.center()[0]))
        self.click(slot, after_sleep=.9)
        boxes = self._read()
        if self._label(boxes, "是否确认覆盖此存档位", (.30, .36, .70, .59), contains=True):
            confirm = self._label(boxes, "确认", (.58, .75, .82, .90))
            if not confirm:
                raise RuntimeError("未识别到空白栏位的保存确认按钮。")
            self.click(confirm, after_sleep=.9)
            boxes = self._read()
        x, y = slot.center()
        saved_card = (x / self.width - .19, y / self.height - .08,
                      x / self.width + .19, y / self.height + .065)
        if not self._label(boxes, "12/02", saved_card):
            raise RuntimeError("未确认空白栏位已写入 12/02 手动档。")
        self.log_info("已在空白栏位保存 12/02 结局前手动档")
        self.click(.954, .068, after_sleep=.65)
        if self.wait_until(lambda: self._settings(self._read()), time_out=5):
            self.click(.954, .068, after_sleep=.65)

    def _forum(self, boxes):
        return self._label(boxes, "SOLAR-X", (.12, .045, .29, .1), contains=True)

    def _full_choice(self, boxes):
        if (self._on_map(boxes)
                or self._label(boxes, "阶段目标", (.72, .11, .91, .22), contains=True)):
            return None
        key = self._label(boxes, "F", (.4, .28, .53, .88))
        options = [b for b in boxes if .54 < b.center()[0] / self.width < .90
                   and .28 < b.center()[1] / self.height < .9
                   and re.search(r"[\u4e00-\u9fff\d]", b.name)]
        if not options:
            return None
        selected = self._selected_choice(options, key, .422, .45, .052)
        if not selected:
            return None
        if answer := self._exam_answer(options, self._date(boxes)):
            return answer
        for title in ("理事会组织的学院开放日", *STORY_CHOICES):
            if option := self._label(options, title, contains=True):
                _, y = option.center()
                half = round(.009 * self.height)
                fill = self.frame[y - half:y + half, round(.54 * self.width):round(.575 * self.width)]
                if np.mean(fill) > 165:
                    return option
        return selected

    def _letter(self, boxes):
        return any(.47 < b.center()[0] / self.width < .53 and .875 < b.center()[1] / self.height < .93
                   and LETTER_PAGE.fullmatch(b.name) for b in boxes)

    def _letter_button(self, boxes):
        return (self._label(boxes, "下一页", (.83, .90, .98, .98), contains=True)
                or self._label(boxes, "结束", (.83, .90, .98, .98)))

    def _envelope(self, boxes):
        return (self._label(boxes, "致:", (.18, .22, .38, .33), contains=True)
                or self._label(boxes, "致：", (.18, .22, .38, .33), contains=True))

    def _black_narration(self, boxes):
        if np.mean(self.frame) >= 25:
            return False
        return any(.15 < b.center()[0] / self.width < .85 and .3 < b.center()[1] / self.height < .7
                   and re.search(r"[\u4e00-\u9fff]", b.name) for b in boxes)

    def _commission_offer(self, boxes):
        if self._label(boxes, "特别委托", (.3, .13, .5, .24)):
            return self._label(boxes, "接取委托", (.5, .85, .8, .97))
        return None

    def _commission_unmet(self, boxes):
        for box in boxes:
            x, y = box.center()
            if .38 < x / self.width < .88 and .45 < y / self.height < .6:
                if match := re.fullmatch(r"(\d+)\s*[/／]\s*(\d+)", box.name.replace(" ", "")):
                    if int(match[1]) < int(match[2]):
                        return True
        return False

    def _attribute_check(self, boxes):
        if not self._label(boxes, "我的属性", (.81, .105, .96, .19), contains=True):
            return None
        options = [b for b in boxes if .14 < b.x / self.width < .435
                   and .62 < b.center()[1] / self.height < .891
                   and re.search(r"[\u4e00-\u9fff]", b.name)]
        for phrase in ("推理", "知识", "思考"):
            if option := self._label(options, phrase, contains=True):
                return option
        key = self._label(boxes, "F", (.126, .61, .195, .9))
        selected = self._selected_choice(options, key, .175, .197, .05)
        return selected or next(iter(options), None)

    def _attribute_result(self, boxes):
        button = self._label(boxes, "完成", (.43, .805, .58, .91))
        if button:
            paper = self.frame[round(.49 * self.height):round(.74 * self.height),
                               round(.38 * self.width):round(.64 * self.width)]
            if np.mean(paper > 180) > .78:
                return button
        return None

    def _lottery_screen(self, boxes):
        return self._label(boxes, "特别刮奖", (.76, .115, .995, .2), contains=True)

    def _shop_screen(self, boxes):
        return (self._label(boxes, "学院商店", (0, 0, .2, .12))
                and self._label(boxes, "商店刷新", (.85, .14, .99, .22), contains=True))

    def _buy_in_shop(self, boxes):
        slots = ((.30, .68), (.30, .40), (.42, .40), (.54, .40), (.66, .40))
        if self._shop_buys < 3:
            self.click(*slots[self._shop_buys % len(slots)], after_sleep=.35)
            updated = self._read()
            if buy := self._label(updated, "购买", (.75, .84, .95, .94)):
                self.click(buy, after_sleep=.7)
                popup = self._read()
                if confirm := (self._label(popup, "确认", (.43, .7, .8, .92))
                               or self._label(popup, "确定", (.32, .69, .72, .96))):
                    self.click(confirm, after_sleep=.6)
                elif (self._label(popup, "获得", (.4, .18, .6, .4))
                      or self._label(popup, "点击空白处继续", (.36, .82, .64, .96))):
                    self.click(.5, .89, after_sleep=.5)
                self._shop_buys += 1
                return
            self._need_shop_buy = False
        self._need_shop_buy = False
        self.click(.954, .067, after_sleep=.65)

    def _buy_gifts(self):
        def gift_receipt():
            current = self._read()
            return current if self._label(current, "新增物品", (.06, .20, .34, .42), contains=True) else None

        # The shop moves a purchased card to the end of the list.
        for _ in range(2):
            self.click(.305, .40, after_sleep=.55)
            boxes = self._read()
            if not self._label(boxes, "送礼物", (.74, .31, .97, .43), contains=True):
                break
            if not (buy := self._label(boxes, "购买", (.75, .84, .96, .95))):
                break
            self.click(buy, after_sleep=.75)
            boxes = self._read()
            if confirm := self._label(boxes, "确认", (.55, .74, .87, .93)):
                self.click(confirm, after_sleep=.85)
            boxes = self.wait_until(gift_receipt, time_out=3)
            if not boxes:
                self.log_warning("未能确认学院商店的礼物购买结果。")
                break
            self._gifts_available += 1
            if not (close := self._label(boxes, "确定", (.40, .76, .60, .92))):
                raise RuntimeError("未识别到礼物购买提示的关闭按钮。")
            self.click(close, after_sleep=.65)
        self._gift_shop_date = None
        self.click(.954, .067, after_sleep=.65)

    def _give_gift(self, button):
        self.click(button, after_sleep=.7)
        boxes = self._read()
        if not self._label(boxes, "角色礼物", (.43, .16, .70, .30), contains=True):
            return
        if use := self._label(boxes, "使用", (.66, .81, .93, .96)):
            self.click(use, after_sleep=.85)
            self._gifts_available -= 1
            boxes = self._read()
        if self._label(boxes, "角色礼物", (.43, .16, .70, .30), contains=True):
            self.click(.954, .067, after_sleep=.65)

    def _play_lottery(self):
        icon = self.frame[round(.044 * self.height):round(.091 * self.height),
                          round(.788 * self.width):round(.833 * self.width)]
        hue, saturation, value = cv2.split(cv2.cvtColor(icon, cv2.COLOR_BGR2HSV))
        colored = (saturation > 115) & (value > 150)
        # The balance switches from a red ticket to a gold coin after the last free draw.
        if (np.mean(colored & ((hue < 10) | (hue > 170))) < .02
                or self._lottery_index >= 28):
            self._finished_lottery = True
            self.click(.942, .067, after_sleep=.75)
            return
        index = self._lottery_index
        self._lottery_index += 1
        self.click(.231 + .0683 * (index % 7), .38 + .121 * (index // 7),
                   name="使用活动兑奖券", after_sleep=1.1)

    def _paper_choice(self, boxes):
        if (self._on_map(boxes)
                or self._label(boxes, "阶段目标", (.72, .11, .91, .22), contains=True)):
            return None
        key = self._label(boxes, "F", (.7, .5, .755, .88))
        choices = [b for b in boxes if .28 < b.x / self.width < .67
                   and .53 < b.center()[1] / self.height < .88
                   and re.search(r"[\u4e00-\u9fff\d]", b.name)]
        selected = self._selected_choice(choices, key, .271, .286, .047)
        if selected:
            return self._exam_answer(choices, self._date(boxes)) or selected
        return None

    def _selected_choice(self, options, key, left, right, half_height):
        if key:
            return next((b for b in options
                         if abs(b.center()[1] - key.center()[1]) < .026 * self.height), None)
        if len(options) < 2:
            return None
        x1, x2 = round(left * self.width), round(right * self.width)
        half = round(half_height * self.height)
        for option in options:
            _, y = option.center()
            hsv = cv2.cvtColor(self.frame[y - half:y + half, x1:x2], cv2.COLOR_BGR2HSV)
            gold = (hsv[:, :, 0] > 18) & (hsv[:, :, 0] < 40) & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 215)
            if np.mean(np.any(gold, axis=1)) > .55:
                return option
        return None

    def _exam_answer(self, options, date):
        for pattern in EXAM_ANSWERS.get(date or self._active_date, ()):
            if answer := next((b for b in options if re.search(pattern, re.sub(r"\s+", "", b.name))), None):
                return answer
        return None

    def _dialogue_panel(self, boxes):
        frame = self.frame
        panel = frame[round(.95 * self.height):round(.975 * self.height),
                      round(.755 * self.width):round(.785 * self.width)]
        levels = panel.mean(axis=2)
        return (np.mean(levels > 155) > .70 and np.mean(levels < 90) > .015
                and any(.24 < b.x / self.width < .76 and .85 < b.center()[1] / self.height < .94
                        and re.search(r"[\u4e00-\u9fff]", b.name) for b in boxes))

    def reset_to_start(self):
        # The activity's current journey is the baseline, including in-progress saves.
        self._claim_log = {}
        self._need_shop_buy = False
        self._shop_buys = 0
        self._journey_number = None
        self._reset_journey_progress()

    def _reset_journey_progress(self):
        self._friend_levels = {}
        self._reward_checkpoints = set()
        self._story_ending = False
        self._schedule_tabs = {}
        self._reused_schedule_dates = set()
        self._gift_shop_dates = set()
        self._gift_visits = set()
        self._gift_shop_date = None
        self._gifts_available = 0
        self._active_date = None
        self._starting_semester = False
        self._finished_lottery = False
        self._lottery_index = 0
        self._skipped_commissions = set()
        self._too_tired_rooms = set()
        self._current_room = None

    def verify_start_state(self):
        return bool(self.wait_until(lambda: self._resume_state(self._read()), time_out=8))

    def _reward_page(self, title):
        return self.ocr(.035, .035, .18, .09, match=title)

    def _settings(self, boxes):
        return self._label(boxes, "学院手册", (.30, .12, .70, .27), contains=True) and self._label(
            boxes, "退出游戏", (.52, .81, .71, .94))

    def _reward_counts(self, boxes):
        for box in boxes:
            if box.x < .15 * self.width and (match := REWARD_PROGRESS.fullmatch(box.name)):
                earned, total = map(int, match.groups())
                self._claim_log[REWARD_PAGES[total - 14]] = f"{earned}/{total}"

    def _all_rewards_earned(self):
        return all(self._claim_log.get(title) == f"{total}/{total}"
                   for title, total in zip(REWARD_PAGES, (14, 15)))

    def _claim_rewards(self, title, button):
        opened = False
        for attempt in range(3):
            if attempt > 0:
                boxes = self._read()
                if self._label(boxes, "档案管理", (0, 0, .2, .12), contains=True):
                    self.click(.954, .068, after_sleep=.65)
                    boxes = self._read()
                elif self._tutorial(boxes):
                    boxes = self._read()
                button = self._label(boxes, title, (0, .42, .2, .9)) or button
            self.click(button, after_sleep=.55)
            if self.wait_until(lambda: self._reward_page(title), time_out=2.5):
                opened = True
                break
        if not opened:
            raise RuntimeError(f"无法打开{title}。")

        for _ in range(8):
            boxes = self._read()
            if (self._label(boxes, "获得", (.45, .19, .55, .36)) or
                    self._label(boxes, "点击空白处继续", (.36, .82, .64, .96))):
                self.click(.5, .89, after_sleep=.5)
                continue
            if claim := self._label(boxes, "领取奖励", (.81, .13, .95, .95)):
                self.click(claim, after_sleep=.65)
                continue
            if not self._label(boxes, title, (0, 0, .2, .13)):
                self.sleep(.4)
                continue
            status = self.ocr(.8, .175, .95, .29)
            roi = self.frame[round(.19 * self.height):round(.27 * self.height),
                             round(.84 * self.width):round(.92 * self.width)]
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            checked = np.mean((hsv[:, :, 0] >= 35) & (hsv[:, :, 0] <= 75)
                              & (hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 150)) > .015
            if not checked and not any(re.search(r"进行中|已领取|已完成", b.name) for b in status):
                self.sleep(.4)
                continue
            if title == "限时奖励":
                self._need_shop_buy = bool(self._label(
                    boxes, "累计购买商品", (.35, .15, .75, .80), contains=True))
            # Available tasks sort first, and one button claims all on that page.
            self.click(.954, .067, after_sleep=.7)
            return
        raise RuntimeError(f"无法确认{title}的领取状态。")

    def _collect_rewards(self, boxes, parent):
        self._reward_counts(boxes)
        for title, total in zip(REWARD_PAGES, (14, 15)):
            if self._claim_log.get(title) == f"{total}/{total}":
                continue
            button = self._label(boxes, title, (0, .42, .2, .9))
            if button:
                self._claim_rewards(title, button)
                boxes = None
                for _ in range(3):
                    boxes = self.wait_until(lambda: (frame if parent(frame := self._read())
                                                     else None), time_out=2.5)
                    if boxes:
                        self.sleep(.35)
                        settled = self._read()
                        if parent(settled):
                            boxes = settled
                        break
                    if self._reward_page(title):
                        self.click(.954, .067, after_sleep=.7)
                if not boxes:
                    raise RuntimeError("领取奖励后无法返回活动菜单。")
                self._reward_counts(boxes)
        if self._claim_log:
            self.info_set("奖励进度", dict(self._claim_log))
        return boxes

    def _dialogue(self, boxes):
        if self._label(boxes, "是否快速跳过剧情", (.23, .3, .8, .58), contains=True):
            if self._label(boxes, "本次登录不再提醒"):
                self.click(.431, .492, after_sleep=.12)
            self.click(.694, .832, after_sleep=.6)
            return True
        if self._letter_button(boxes):
            self.click(.894, .872, after_sleep=.6)
            return True
        options = [b for b in boxes if b.x > .67 * self.width
                   and .25 < b.y / self.height < .73 and re.search(r"[\u4e00-\u9fff]", b.name)]
        key = self._label(boxes, "F", (.65, .25, .7, .76))
        panel = self._dialogue_panel(boxes)
        if options and (key or (panel and self.ocr(.78, .05, .83, .11, match="Tab"))):
            if answer := self._exam_answer(options, self._date(boxes)):
                self.click(answer, after_sleep=.65)
                return True
            for phrase in STORY_CHOICES:
                if option := self._label(options, phrase, contains=True):
                    self.click(option, after_sleep=.65)
                    return True
            self.click(options[0], after_sleep=.65)
            return True
        if self._label(boxes, "长按加速", (.83, .88, 1, .98), contains=True):
            self.click(.954, .068, after_sleep=.55)
            return True
        if self._black_narration(boxes):
            self.click(.954, .068, after_sleep=.55)
            return True
        if panel:
            self.click(.77, .955, after_sleep=.45)
            return True
        return False

    def _on_map(self, boxes):
        return bool(self._label(boxes, "人背包", (.48, .02, .66, .13), contains=True))

    def _map_marker(self, boxes):
        # Red exclamation badges are map events. OCR supplies the map context,
        # while the icon remains readable when a building label is obscured.
        frame = self.frame
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        mask = ((hue < 8) | (hue > 170)) & (saturation > 145) & (value > 175)
        _, _, stats, centers = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        directions = ()
        for (x, y, w, h, area), (cx, cy) in zip(stats[1:], centers[1:]):
            rx, ry = cx / self.width, cy / self.height
            if not (.025 < rx < .985 and .07 < ry < .96
                    and .011 < w / self.width < .042 and .018 < h / self.height < .07
                    and .58 < w / h < 1.45 and area > .35 * w * h):
                continue
            red_icon = hsv[y:y + h, x:x + w]
            white = (red_icon[:, :, 1] < 65) & (red_icon[:, :, 2] > 195)
            if np.count_nonzero(white) <= .07 * w * h:
                continue
            for box in boxes:
                tx, ty = box.center()
                if tx < .20 * self.width and ty > .60 * self.height:
                    continue
                if tx / self.width > .71 and ty / self.height < .36:
                    continue
                if any(word in box.name for word in ("阶段目标", "距离", "还有")):
                    continue
                if (0 < cx - tx < .09 * self.width and 0 < ty - cy < .065 * self.height
                        and 2 <= len(box.name) <= 10 and re.search(r"[\u4e00-\u9fff]", box.name)):
                    return box, ()
            if w / self.width >= .018:
                horizontal = "a" if rx < .22 else "d" if rx > .84 else ""
                vertical = "w" if ry < .185 else "s" if ry > .865 else ""
                if horizontal or vertical:
                    directions = tuple(horizontal + vertical)
        return None, directions

    def _enter_location(self, location):
        self.click(location, after_sleep=.75)
        boxes = self._read()
        if self._on_map(boxes):
            same = self._label(boxes, location.name, (.06, .2, .93, .97))
            if same:
                x, y = same.center()
                self.click(int(x - .02 * self.width), int(y - .11 * self.height), after_sleep=.65)

    def _tutorial(self, boxes):
        if direction := self._label(boxes, "前往【", (.1, .2, .9, .79), contains=True):
            x, y = direction.center()
            self.click(x / self.width, min(.94, y / self.height + .17), after_sleep=.7)
            return True
        for text, point in (
            ("点击【背包】", (.247, .86)),
            ("使用【日常补给】", (.786, .886)),
            ("在当前周目结束后不会消失", (.58, .45)),
            ("剧情选择会影响属性培养方向", (.115, .77)),
            ("此处会显示当前历程的", (.938, .234)),
            ("可以邀请伙伴外出或赠送礼物", (.326, .5)),
            ("可进行【档案存储】", (.954, .067)),
            ("点击【档案存储】", (.382, .808)),
            ("可查看个人信息与角色好感度", (.246, .68)),
            ("角色会发布【限时】任务", (.655, .49)),
            ("需要连续占用多天的日程", (.516, .27)),
            ("执行【休息】类行为后", (.773, .15)),
        ):
            if self._label(boxes, text, contains=True):
                self.click(*point, after_sleep=.65)
                return True
        if (self._label(boxes, "A", (.22, .75, .30, .86))
                and self._label(boxes, "D", (.70, .75, .78, .86))):
            self.click(.963, .225, after_sleep=.55)
            return True
        return False

    def _stage_goal(self, boxes):
        if not self._label(boxes, "阶段目标", (.72, .11, .91, .22), contains=True):
            return False
        frame = self.frame
        timer = self._label(boxes, "还有", (.75, .27, .94, .55), contains=True)
        bottom = timer.center()[1] / self.height - .035 if timer else .29
        area = frame[round(.17 * self.height):round(bottom * self.height),
                     round(.93 * self.width):round(.97 * self.width)]
        hsv = cv2.cvtColor(area, cv2.COLOR_BGR2HSV)
        badge = ((hsv[:, :, 0] < 23) | (hsv[:, :, 0] > 170)) & (hsv[:, :, 1] > 145)
        badge &= hsv[:, :, 2] > 155
        count, _, stats, centers = cv2.connectedComponentsWithStats(badge.astype(np.uint8))
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] > self.width * self.height * .00004:
                x, y = centers[index]
                self.click(int(x + .927 * self.width), int(y + .195 * self.height), after_sleep=.65)
                return True
        return False

    def _fatigue(self, boxes):
        for box in boxes:
            x, y = box.center()
            if .57 < x / self.width < .77 and y / self.height < .11:
                if match := re.fullmatch(r"(\d{1,3})\s*[%％]", box.name):
                    return int(match[1])
        return None

    def _stat(self, boxes, name):
        label = self._label(boxes, name, (.035, .63, .1, .95), contains=True)
        if label:
            return next((int(b.name) for b in boxes if .12 < b.x / self.width < .18
                         and abs(b.center()[1] - label.center()[1]) < .02 * self.height
                         and b.name.isdecimal()), None)
        return None

    def _course_order(self, boxes):
        knowledge = self._stat(boxes, "知识")
        art = self._stat(boxes, "艺术")
        physique = self._stat(boxes, "体魄")
        if knowledge is None or knowledge < 65:
            return ("理科教室", "文科教室", "操场", "家庭餐厅")
        if art is not None and art < 70:
            return ("文科教室", "操场", "理科教室", "家庭餐厅")
        if physique is not None and physique < 35:
            return ("操场", "理科教室", "文科教室", "家庭餐厅")
        if knowledge < 850:
            return ("理科教室", "家庭餐厅", "文科教室", "操场")
        return ("家庭餐厅", "理科教室", "操场", "文科教室")

    def _visit_location(self, boxes):
        if (self._label(boxes, "居家休息", (.7, .4, .99, .98), contains=True)
                or self._label(boxes, "日程安排", (.7, .87, .87, .98), contains=True)
                or self._label(boxes, "阶段目标", (.72, .11, .91, .22), contains=True)):
            return False
        room = next((name for name in (*COURSES, "街道")
                     if self._label(boxes, name, (.02, .14, .23, .23))), None)
        slot_tag = self._label(boxes, "占用「", (.72, .2, .99, .85), contains=True)
        if not room and not slot_tag and not self._label(boxes, "赠送礼物", (.19, .3, .4, .46), contains=True):
            return False
        if room:
            self._current_room = room
        if self._label(boxes, "值过高", (.4, .7, .65, .85), contains=True):
            if room:
                self._too_tired_rooms.add(room)
            self.click(.954, .068, after_sleep=.65)
            return True
        buying_gifts = self._gift_shop_date is not None and self._gift_shop_date == self._active_date
        if room == "学院商店" and (buying_gifts or (self._need_shop_buy and self._shop_buys < 3)):
            if shop := self._label(boxes, "学院商店", (.70, .35, .95, .46)):
                self.click(shop, after_sleep=.75)
                return True
        if lottery := self._label(boxes, "特别刮奖", (.7, .44, .98, .65), contains=True):
            if self._finished_lottery:
                self.click(.954, .068, after_sleep=.7)
            else:
                self.click(lottery, after_sleep=.85)
            return True
        for level, text in enumerate(("初遇", "熟识", "相知", "挚友")):
            if room and self._label(boxes, text, (.7, .125, .93, .205), contains=True):
                self._friend_levels[room] = level
                break
        visit = (self._active_date, room)
        if self._gifts_available and room in GIFT_LOCATIONS and visit not in self._gift_visits:
            self._gift_visits.add(visit)
            if gift := self._label(boxes, "送礼物", (.17, .30, .45, .47), contains=True):
                self._give_gift(gift)
                return True
        commission = self._label(boxes, "特别委托", (.19, .23, .96, .8))
        if commission and (self._active_date, room) in self._skipped_commissions:
            commission = None
        invite = self._label(boxes, "邀约出行", (.2, .3, .52, .66), contains=True)
        if action := commission or invite:
            self.click(action, after_sleep=.65)
            return True
        fatigue = self._fatigue(boxes)
        if fatigue is None or fatigue < 88:
            for text in COURSES.get(room, ()):
                if action := self._label(boxes, text, (.67, .22, .95, .87), contains=True):
                    self.click(action, after_sleep=.65)
                    return True
            if slot_tag:
                x, y = slot_tag.center()
                self.click(int(x - .08 * self.width), int(y + .04 * self.height), after_sleep=.65)
                return True
        if room and fatigue is not None and fatigue >= 88:
            self._too_tired_rooms.add(room)
        self.click(.954, .068, after_sleep=.65)
        return True

    def _schedule_save(self, boxes):
        if self._label(boxes, "疲劳度预览", (.53, .035, .64, .112), contains=True):
            return (self._label(boxes, "执行日程", (.78, .85, .99, .98))
                    or self._label(boxes, "保存日程", (.78, .85, .99, .98)))
        return None

    def _schedule_stat(self, boxes, x):
        for box in boxes:
            cx, cy = box.center()
            if abs(cx / self.width - x) < .017 and .88 < cy / self.height < .94:
                value = box.name.replace(",", "")
                if value.isdecimal():
                    return int(value)
        return 0

    def _schedule_money(self, boxes):
        for box in boxes:
            cx, cy = box.center()
            if .8 < cx / self.width < .86 and .035 < cy / self.height < .11:
                value = box.name.replace(",", "")
                if value.isdecimal():
                    return int(value)
        return None

    def _schedule_activity(self, tab, activities):
        self.click(tab, .15, after_sleep=.45)
        cards = self._read()
        if self._tutorial(cards):
            self.click(tab, .15, after_sleep=.45)
            cards = self._read()
        for activity in activities:
            if card := self._label(cards, activity, (.66, .2, .83, .74), contains=True):
                _, y = card.center()
                self.click(round(.896 * self.width), y, after_sleep=.55)
                return True
        return False

    def _commission_tabs(self, date):
        if date in self._schedule_tabs:
            return self._schedule_tabs[date]
        self.click(.654, .15, after_sleep=.45)
        boxes = self._read()
        if self._tutorial(boxes):
            boxes = self._read()
        tab = self.frame[round(.135 * self.height):round(.177 * self.height),
                         round(.64 * self.width):round(.67 * self.width)]
        hsv = cv2.cvtColor(tab, cv2.COLOR_BGR2HSV)
        gold = (hsv[:, :, 0] >= 13) & (hsv[:, :, 0] <= 33) & (hsv[:, :, 1] > 140)
        has_commissions = np.mean(gold) > .10 or bool(self._label(
            boxes, "待执行次数", (.42, .36, .61, .52), contains=True))
        if date is not None:
            self._schedule_tabs[date] = has_commissions
        if has_commissions:
            self._schedule_commission(boxes)
        return has_commissions

    def _schedule_commission(self, boxes):
        if not self._label(boxes, "周一", (.045, .035, .24, .09), contains=True):
            return
        cards = [b for b in boxes if .68 < b.x / self.width < .85 and .21 < b.y / self.height < .7
                 and len(re.findall(r"[\u4e00-\u9fff]", b.name)) >= 4]
        daytime = (.11, .24, .25, .7)
        if any(self._label(boxes, card.name, daytime) for card in cards):
            return
        for card in cards:
            _, y = card.center()
            self.click(round(.895 * self.width), y, name=card.name, after_sleep=.65)
            updated = self._read()
            if any(self._label(updated, choice.name, daytime) for choice in cards):
                return

    def _schedule_step(self, boxes, save, date):
        date = date or self._active_date
        slots = [b for b in boxes if .11 < b.x / self.width < .37
                 and .235 < b.center()[1] / self.height < .7 and "暂无安排" in b.name]
        if not slots:
            self.click(save, after_sleep=.8)
            return

        fatigue = self._fatigue(boxes)
        second_journey = self._journey_number == 2
        if (date and date > (4, 8) and len(slots) == 5
                and fatigue is not None and fatigue <= 30 and date not in self._reused_schedule_dates):
            if reuse := self._label(boxes, "沿用上周", (.58, .86, .86, .98), contains=True):
                self._reused_schedule_dates.add(date)
                self.click(reuse, after_sleep=.65)
                return
        offset = .059 if self._commission_tabs(date) else 0
        if (len(slots) == 5 and self._label(boxes, "04/01", (.045, .03, .22, .095), contains=True)
                and self._schedule_activity(.654, ("文科课程",))):
            return
        self.click(slots[0], after_sleep=.18)
        if fatigue is None:
            fatigue = self._fatigue(self._read())
        if fatigue is None:
            raise RuntimeError("无法读取日程中的疲劳度预览。")
        if fatigue >= 70:
            if self._schedule_activity(.773 + offset, ("居家休息",)):
                return

        knowledge = self._schedule_stat(boxes, .215)
        thinking = self._schedule_stat(boxes, .284)
        art = self._schedule_stat(boxes, .356)
        physique = self._schedule_stat(boxes, .426)
        money = self._schedule_money(boxes)
        science = COURSES["理科教室"]
        work = COURSES["家庭餐厅"]
        if second_journey and self._schedule_activity(.713 + offset, work):
            return
        study_plan = (.23, .235, .37, .7)
        needs_badges = (knowledge >= 65 and money is not None and money < 20000
                        and not any(self._label(boxes, job, study_plan, contains=True) for job in work))
        if (needs_badges or (knowledge >= 850 and thinking >= 600)) and self._schedule_activity(
                .713 + offset, work):
            return
        if knowledge < 65:
            activities = science
        elif physique < 35:
            activities = COURSES["操场"] + science
        elif art < 70:
            activities = COURSES["文科教室"] + science
        else:
            activities = science
        if self._schedule_activity(.654 + offset, activities):
            return
        raise RuntimeError("找不到可加入日程的课程。")

    def _play_step(self, boxes, date):
        if self._tutorial(boxes):
            return True
        if (self._journey_number == 2
                and self._label(boxes, "即将进入结局阶段", (.29, .38, .73, .59), contains=True)):
            if confirm := self._label(boxes, "确认", (.43, .76, .58, .94)):
                self._save_ending_checkpoint(confirm)
                return True
        if self._label(boxes, "本次共获取", (.5, .2, .95, .98), contains=True):
            self.click(.954, .068, after_sleep=.7)
            return True
        if ending := self._ending_selector(boxes):
            if self._label(boxes, "已达成该结局", (.40, .80, .60, .90), contains=True):
                self.click(.812, .378, after_sleep=.75)
            else:
                self.click(ending, after_sleep=1.2)
            return True
        if self._label(boxes, "顺利通关", (.20, .72, .80, .97), contains=True):
            self._story_ending = True
        if self._label(boxes, "故事结局", (.14, .05, .86, .65), contains=True):
            for label in ("返回标题", "返回首页", "返回主界面"):
                if button := self._label(boxes, label, (.25, .7, .99, .98), contains=True):
                    self._story_ending = True
                    self.click(button, after_sleep=1)
                    return True
        if next_month := self._label(boxes, "进入下个月", (.75, .85, .96, .95)):
            self.click(next_month, after_sleep=.8)
            return True
        if self._label(boxes, "是否仍要继续接取", (.22, .35, .83, .56), contains=True):
            self._skipped_commissions.add((date or self._active_date, self._current_room))
            self.click(.31, .833, after_sleep=.65)
            return True
        if offer := self._commission_offer(boxes):
            key = (date or self._active_date, self._current_room)
            if self._commission_unmet(boxes):
                self._skipped_commissions.add(key)
                self.click(.954, .068, after_sleep=.65)
            else:
                self.click(offer, after_sleep=.75)
                if self._commission_offer(self._read()):
                    self._skipped_commissions.add(key)
                    self.click(.954, .068, after_sleep=.65)
            return True
        if option := self._attribute_check(boxes):
            self.click(option, after_sleep=.75)
            return True
        if result := self._attribute_result(boxes):
            self.click(result, after_sleep=.65)
            return True
        if self._lottery_screen(boxes):
            self._play_lottery()
            return True
        if self._shop_screen(boxes):
            if self._gift_shop_date is not None and self._gift_shop_date == self._active_date:
                self._buy_gifts()
            else:
                self._buy_in_shop(boxes)
            return True
        if save := self._schedule_save(boxes):
            self._schedule_step(boxes, save, date)
            return True
        if self._settings(boxes):
            self._collect_rewards(boxes, self._settings)
            for _ in range(3):
                self.click(.954, .067, after_sleep=.8)
                if self.wait_until(lambda: not self._settings(self._read()), time_out=2):
                    return True
            raise RuntimeError("无法关闭学院手册。")
        if self._label(boxes, "角色档案", (0, 0, .2, .12), contains=True):
            self.click(.954, .067, after_sleep=.65)
            return True
        if self._label(boxes, "档案管理", (0, 0, .2, .12), contains=True):
            self.click(.954, .068, after_sleep=.65)
            return True
        if any(self._label(boxes, title, (0, 0, .2, .13)) for title in REWARD_PAGES):
            self.click(.954, .067, after_sleep=.7)
            return True
        if self._label(boxes, "历程已结束", (0, .45, 1, .85), contains=True):
            self.click(.954, .068, after_sleep=.65)
            return True
        if self._forum(boxes):
            self.click(.81, .876, after_sleep=.3)
            self.click(.857, .095, after_sleep=.55)
            return True
        if self._label(boxes, "课题清单", (0, 0, .2, .12), contains=True):
            claim = self._label(boxes, "领取", (.8, .16, .95, .94), contains=True)
            if claim:
                self.click(claim, after_sleep=.65)
            else:
                self.click(.954, .068, after_sleep=.65)
            return True
        if self._label(boxes, "倍速播放", (.86, .89, .99, .99), contains=True):
            self.send_key("ctrl", down_time=1.4, after_sleep=.25)
            return True
        if focus := self._study_focus(boxes):
            self.click(focus, after_sleep=.65)
            return True
        if option := self._full_choice(boxes):
            self.click(option, after_sleep=.65)
            return True
        if option := self._paper_choice(boxes):
            self.click(option, after_sleep=.65)
            return True
        if execute := self._label(boxes, "确认执行", (.35, .55, .74, .96)):
            self.click(execute, after_sleep=.65)
            return True
        if confirm := self._confirm_event(boxes):
            self.click(confirm, after_sleep=.6)
            return True
        confirm = self._label(boxes, "确定", (.32, .69, .72, .96))
        confirm = confirm or self._label(boxes, "确认", (.43, .77, .57, .92))
        if confirm:
            self.click(confirm, after_sleep=.55)
            return True
        if self._dialogue(boxes):
            return True
        if self._envelope(boxes):
            self.click(.5, .58, after_sleep=.8)
            return True
        if self._label(boxes, "新增物品", (.06, .2, .33, .4), contains=True):
            self.send_key("space", after_sleep=.55)
            return True
        if self._label(boxes, "日常补给") and self._label(boxes, "珍藏道具"):
            self.click(.954, .067, after_sleep=.55)
            return True
        if not date:
            return False
        weekend = (any(self._label(boxes, day, (.035, .085, .11, .14)) for day in ("周六", "周日"))
                   or self._label(boxes, "周末可邀请伙伴", (.025, .13, .31, .24), contains=True))
        schedule = self._label(boxes, "日程安排", (.7, .87, .87, .98), contains=True)
        on_map = self._on_map(boxes)
        outing_btn = next((b for b in boxes if .83 <= b.center()[0] / self.width <= .99
                           and .88 <= b.center()[1] / self.height <= .98
                           and re.search(r"[\u4e00-\u9fff]", b.name)
                           and "回家" not in b.name), None)
        event = (any(self._label(boxes, label, (.83, .88, .99, .98), contains=True) for label in OUTINGS)
                 or (not on_map and bool(self._label(boxes, "就是今天啦", (.72, .25, .96, .45), contains=True)))
                 or (not on_map and bool(schedule) and bool(outing_btn) and "外出" not in outing_btn.name))
        outing = event or bool(self._label(boxes, "外出", (.83, .88, .99, .98), contains=True))
        home = self._label(boxes, "回家", (.82, .88, .99, .98), contains=True)
        if schedule or outing or home:
            checkpoint = (date[0], (date[1] - 1) // 7) if date >= (3, 25) else None
            if checkpoint and checkpoint not in self._reward_checkpoints:
                self._reward_checkpoints.add(checkpoint)
                self.click(.954, .067, after_sleep=.65)
                return True
        if schedule and not (event or weekend or self._label(boxes, "本日暂无法进行", contains=True)):
            self.click(.793, .855, after_sleep=.65)
            return True
        if not on_map and self._stage_goal(boxes):
            return True
        if self._visit_location(boxes):
            return True
        if (not weekend or self._too_tired_rooms) and (rest := self._label(boxes, "居家休息", (.7, .4, .99, .98), contains=True)):
            if (self._fatigue(boxes) or 0) >= 75:
                self.click(rest, after_sleep=.65)
                return True
        if outing:
            self.click(.908, .919, after_sleep=.75)
            return True
        if on_map:
            if home and not weekend and (self._fatigue(boxes) or 0) >= 85:
                self.click(.918, .868, after_sleep=.65)
                return True
            if self._journey_number == 2 and weekend and self._enter_gift_location(boxes, date):
                return True
            marker, directions = self._map_marker(boxes)
            if marker:
                self._enter_location(marker)
                return True
            if directions:
                for key in directions:
                    self.send_key(key, down_time=.55, after_sleep=.17)
                return True
            if not home:
                for label in OUTINGS:
                    for box in boxes:
                        bx, by = box.center()
                        rx, ry = bx / self.width, by / self.height
                        if not (.06 <= rx <= .93 and .18 <= ry <= .92):
                            continue
                        if (rx > .71 and ry < .36) or (rx < .26 and ry < .24):
                            continue
                        if any(word in box.name for word in ("阶段目标", "距离", "还有")):
                            continue
                        if label in re.sub(r"\s+", "", box.name):
                            self._enter_location(box)
                            return True
            order = self._course_order(boxes)
            if weekend:
                order = sorted(COURSES, key=lambda name: self._friend_levels.get(name, 0))
            if self._need_shop_buy and self._shop_buys < 3:
                order = ("学院商店", *(name for name in order if name != "学院商店"))
            for name in order:
                if name in self._too_tired_rooms:
                    continue
                if location := self._label(boxes, name, (.06, .2, .93, .97)):
                    self._enter_location(location)
                    return True
            if home:
                self.click(.918, .868, after_sleep=.65)
                return True
        return False

    def _enter_gift_location(self, boxes, date):
        map_labels = (.06, .20, .95, .97)
        if (date not in self._gift_shop_dates
                and (stock := self._label(boxes, "新品上架", (.01, .10, .31, .21), contains=True))):
            self._gift_shop_dates.add(date)
            self._gift_shop_date = date
            self.log_info("学院商店新品上架，查看可购买的礼物")
            if shop := self._label(boxes, "学院商店", map_labels):
                self._enter_location(shop)
            else:
                self.click(stock, after_sleep=.7)
            return True
        if self._gifts_available:
            for name in GIFT_LOCATIONS:
                if (date, name) not in self._gift_visits:
                    if location := self._label(boxes, name, map_labels):
                        self._enter_location(location)
                        return True
        return False

    def run(self):
        self.reset_to_start()
        if not self.verify_start_state():
            raise RuntimeError("请先打开「第二索拉：未名之梦」活动首页或当前旅程。")

        last_date = None
        completed_journeys = 0
        recent = deque(maxlen=40)
        entered = False
        idle_since = monotonic()
        try:
            for _ in range(10000):
                boxes = self._read()
                date = self._date(boxes)
                if date and date != last_date:
                    if last_date and last_date >= (12, 1) and date[0] == 3 and not self._starting_semester:
                        self._reset_journey_progress()
                        self._journey_number = 2
                        recent.clear()
                    last_date = date
                    self._active_date = date
                    self._starting_semester = False
                    self._finished_lottery = False
                    self._lottery_index = 0
                    self._too_tired_rooms.clear()
                    self.log_info(f"学期进度 {date[0]:02}/{date[1]:02}")
                if not entered and (self._lobby(boxes) or self._archive_page(boxes)):
                    self._start_from_archive(boxes)
                    entered = True
                    idle_since = monotonic()
                    continue
                if self._lobby(boxes):
                    boxes = self._collect_rewards(boxes, self._lobby)
                    resume = self._continue_journey_button(boxes)
                    finished = self._story_ending or (not resume and last_date and last_date >= (12, 1))
                    if finished and self._all_rewards_earned():
                        return dict(self._claim_log)
                    if not finished or completed_journeys >= 2:
                        raise RuntimeError(f"旅程已返回活动首页，奖励进度 {self._claim_log}")
                    completed_journeys += 1
                    self._reset_journey_progress()
                    recent.clear()
                    self._start_from_archive(boxes)
                    idle_since = monotonic()
                    continue
                entered = entered or date is not None
                signature = tuple(re.sub(r"\s+", "", b.name) for b in boxes
                                  if re.search(r"[\u4e00-\u9fff]|\d/\d", b.name))
                transition = (not date and (len(signature) < 3 or self._letter(boxes))) or (
                    date == (3, 18) and self._label(boxes, "迎来开学第一课", contains=True))
                if "长按加速" in signature:
                    stuck = len(recent) >= 15 and all(s == signature for s in list(recent)[-15:])
                else:
                    limit = 35 if self._lottery_screen(boxes) else 7
                    stuck = bool(signature) and recent.count(signature) >= limit
                if stuck:
                    raise RuntimeError(f"活动界面重复出现：{'，'.join(signature[:12])}")
                if self._play_step(boxes, date):
                    if self._story_ending and self._all_rewards_earned():
                        return dict(self._claim_log)
                    recent.append(signature)
                    idle_since = monotonic()
                elif monotonic() - idle_since > (125 if transition else 25):
                    raise RuntimeError(f"无法识别下一步：{'，'.join(signature[:16])}")
                else:
                    self.sleep(.35)
            raise RuntimeError("超过学期操作次数上限。")
        except RuntimeError:
            self.screenshot("galgame_stopped")
            raise
