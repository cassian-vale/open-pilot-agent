# coding: utf-8
class SuffixTree:
    def __init__(self):
        self.common_abbreviations = [
            # 称谓/头衔
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Rev.", "Sr.", "Jr.", "Hon.", "Gen.", "Col.", "Capt.", "Lt.", "Sgt.", "Brig.",
            "Maj.",
            # 地理/国家
            "U.S.", "U.S.A.", "U.K.", "E.U.", "N.Y.", "L.A.", "D.C.", "N.Z.", "S.A.", "W.A.", "E.C.", "W.C.",
            # 时间/日期
            "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
            "A.M.", "P.M.", "B.C.", "A.D.", "Mon.", "Tue.", "Wed.", "Thu.", "Fri.", "Sat.", "Sun.",
            # 组织/机构
            "Inc.", "Co.", "Ltd.", "Corp.", "Univ.", "Inst.", "Assn.", "Gov.", "Dept.", "Div.", "Sec.", "Rep.", "Sen.", "Pres.",
            "Vice-Pres.",
            # 其他常见缩写
            "etc.", "e.g.", "i.e.", "vs.", "a.m.", "p.m.", "No.", "Vol.", "pp.", "ex.", "al.", "viz.", "cf.", "ibid.", "op.",
            "cit.", "id.", "approx.", "max.", "min.", "avg.", "std.", "temp.", "vol.", "ref.", "fig.", "chap.", "ed.", "trans.",
            "publ.", "anon.",
            # 学科/领域
            "Math.", "Phys.", "Chem.", "Biol.", "Eng.", "Lit.", "Hist.", "Arch.", "Ast.", "Bot.", "Zool.", "Geol.", "Met.", "Stat.",
            # 医学/科学
            "DNA", "RNA", "MRI", "CT", "PET", "Hz", "kHz", "MHz", "diag.", "prescr.", "Rx", "Dr.", "Ph.D.", "M.D.", "B.Sc.",
            # 法律/商业
            "vs.", "att.", "cert.", "est.", "inv.", "acct.", "insp.",
            # 技术/工程
            "amp.", "volt.", "watt.", "Hz.", "kHz.", "MHz.", "GHz.",
            # 常见品牌/术语
            "Corp.", "Intl.", "Assn.", "Soc.", "Inst.", "Univ.",
        ]
        self.tree = {}  # 字典树结构
        self.processed_abbr_set = set([abbr for abbr in self.common_abbreviations])
        for abbreviation in self.processed_abbr_set:
            self.insert(abbreviation)
        self.max_len = max([len(abbr) for abbr in self.common_abbreviations])

    def insert(self, word):
        """将单词插入字典树"""
        node = self.tree
        for char in reversed(word):  # 从后向前插入（构建后缀树）
            if char not in node:
                node[char] = {}
            node = node[char]
        node["#"] = True  # 标记单词结束

    def is_suffix(self, word):
        """判断word是否以字典树中的某个词为后缀"""
        node = self.tree
        for char in reversed(word[:self.max_len]):  # 从后向前匹配
            if char not in node:
                return False
            node = node[char]
            if "#" in node:  # 如果匹配到完整的后缀
                return True
        return False

    @staticmethod
    def process_sentence(sentence):
        sentences_ = []
        start_idx, end_idx = 0, 0
        for i, char in enumerate(sentence):
            end_idx = i + 1
            if char in "。？！；":
                if char != "\n" and i < len(sentence) - 1 and sentence[i+1] in "‘’”“'\"[]【】（）(){}":
                    end_idx += 1
                sentences_.append(sentence[start_idx: end_idx])
                start_idx = end_idx
        if end_idx > start_idx:
            sentences_.append(sentence[start_idx: end_idx])
        return sentences_

    def split_to_sentences(self, text):
        words = text.split(" ")
        sentences = []
        sentence = ""
        for i, word in enumerate(words):
            sentence += word
            if i < len(words) - 1:
                sentence += " "
            if word:
                if word[-1] == ".":
                    if word not in self.processed_abbr_set and not self.is_suffix(word):
                        sentences_ = self.process_sentence(sentence)
                        sentences.extend(sentences_)
                        sentence = ""
                elif word[-1] in "!?;":
                    sentences_ = self.process_sentence(sentence)
                    sentences.extend(sentences_)
                    sentence = ""

        if sentence:
            sentences_ = self.process_sentence(sentence)
            sentences.extend(sentences_)
        return sentences
