# coding=utf-8
import sys
import time
from pathlib import Path

dir_name = Path(__file__).resolve().parent.parent
sys.path.append(str(dir_name))


from preprocess.split_sentence import SuffixTree


class TextChunker:
    def __init__(self):
        self.split_tool = SuffixTree()

    def chunk(self, text, chunk_size=512, overlap=100, return_sentences=False):
        chunks = []
        sentences = self.split_tool.split_to_sentences(text)

        if return_sentences:
            return sentences

        if not sentences:
            return chunks

        start = 0
        while start < len(sentences):
            current_chunk = []
            current_length = 0
            end = start

            # 收集不超过 chunk_size 的句子
            while end < len(sentences):
                sentence_length = len(sentences[end])
                if current_length + sentence_length > chunk_size:
                    break
                current_chunk.append(sentences[end])
                current_length += sentence_length
                end += 1

            # 处理单个句子超过 chunk_size 的情况
            if not current_chunk:
                current_chunk = [sentences[start]]
                end = start + 1

            chunks.append(''.join(current_chunk))
            if end == len(sentences):
                break

            if overlap > 0:

                # 计算下一个块的起始位置
                new_start = self._find_overlap_start(sentences, end - 1, chunk_size, overlap)

                # # 避免无限循环，确保起始位置递增
                if new_start <= start:
                    new_start = end
                start = new_start
            else:
                start = end

        return chunks

    def _find_overlap_start(self, sentences, end, chunk_size, overlap):
        if end < 0:
            return 0

        # 如果当前仅剩一个块儿
        if end + 1 == len(sentences) - 1 and len(sentences[end]) + len(sentences[end+1]) > chunk_size:
            print(999, len(sentences[end]), len(sentences[end+1]))
            return end + 1

        total = 0
        start_idx = end

        # 至少包含当前句子（即使超过 overlap）
        total += len(sentences[start_idx])
        start_idx -= 1

        # 向前遍历以找到最佳重叠起始点
        while start_idx >= 0:
            current_length = len(sentences[start_idx])
            if total + current_length > overlap:
                break
            total += current_length
            start_idx -= 1

        return max(start_idx + 1, 0)

    def add_start_end(self, chunks, start=0):
        end = start
        text = ""
        for chunk in chunks:
            if chunk:
                end = start + len(chunk)
                text += f"<start={start}>{chunk}<end={end}>\n"
                start = end
        return text


if __name__ == "__main__":
    text = """
    新华社北京1月17日电（记者潘洁、韩佳诺）国家统计局17日发布数据显示，初步核算，2024年国内生产总值（GDP）1349084亿元，按不变价格计算，比上年增长5.0%。“2024年我国经济克服了复杂的内外环境带来的各种困难和挑战，顺利地实现了主要预期目标任务，推动了经济质的有效提升和量的合理增长，高质量发展成色十足，成绩实属不易。”国家统计局局长康义当天在国新办发布会上说。
    在外部压力增大、内部困难增多的条件下，2024年我国经济总量再上新台阶，首次突破130万亿元，规模稳居全球第二位。从全球看，我国5%的经济增速在世界主要经济体中名列前茅，继续是世界经济增长的重要动力源。
    分季度看，一季度GDP同比增长5.3%，二季度增长4.7%，三季度增长4.6%，四季度增长5.4%。从环比看，四季度GDP增长1.6%。
    康义说，针对2024年二季度、三季度我国经济增速放缓的情况，党中央因时因势加强宏观调控，一揽子政策及时出台，有效提振了社会信心，促进经济明显回升。四季度，我国规模以上工业增加值、服务业增加值、社会消费品零售总额的增速比三季度分别加快0.7、1.0和1.1个百分点。
    过去一年，我国高质量发展取得新成效，保障和改善民生扎实推进，粮食产量创历史新高，重点领域风险有序有效化解。2024年，全国规模以上高技术制造业、装备制造业增加值占规模以上工业增加值的比重分别上升到16.3%、34.6%，比上年分别提高0.6和1.0个百分点。全国城镇调查失业率年均值为5.1%，比上年下降0.1个百分点；居民人均可支配收入实际增长5.1%，与经济增长同步。
    “我们也要清醒认识到，外部环境带来的不利影响在加深，国内需求不足，部分企业生产经营困难，群众就业增收承压，风险隐患仍然较多，推动经济回升向好还需要付出艰苦的努力。”康义说，下阶段，要按照中央经济工作会议的决策部署，正视困难、坚定信心、干字当头，把各方面有利因素转化为发展的实绩，不断推动经济持续向好。
    新华社北京1月17日电（记者张晓洁、张辛欣）国家统计局17日发布数据显示，2024年全国规模以上工业增加值比上年增长5.8%，工业生产增势较好，装备制造业和高技术制造业增长较快。
    在外部压力加大、内部困难增多的复杂严峻形势下，我国工业经济发展实现稳中向好，工业体系全、品种多、规模大的优势进一步巩固。国家统计局局长康义在17日举行的国新办新闻发布会上说，我国制造业规模居世界第一，5G、算力、储能等新型基础设施加快布局，制造业强链补链扎实推进，安全发展基础巩固夯实。
    特别是在一系列政策推动下，工业明显回升。2024年四季度规模以上工业增加值同比增长5.7%，比三季度加快0.7个百分点，“两新”“两重”政策效应释放，规模以上装备制造业增加值增长8.1%，比三季度加快1.1个百分点。
    数据显示，我国全球创新指数2024年排名升到第11位，是十年来创新力提升最快的经济体之一。
    2024年，规模以上高技术制造业增加值比上年增长8.9%，智能车载设备制造、智能无人飞行器制造等行业增加值分别增长25.1%、53.5%。新兴产业进一步壮大，未来产业积极布局。2024年制造业技改投资比上年增长8%，明显快于全部投资的增速，产业转型升级加快。规模以上数字产品制造业增加值增速明显快于规模以上工业，折射了数字经济的进一步成长。
    记者了解到，下一步，我国还将深入实施制造业重点产业链高质量发展行动，提升产业科技创新能力，改造升级传统产业，巩固提升优势产业，培育壮大新兴产业，前瞻布局未来产业，支持中小企业专精特新发展，培育壮大新质生产力，加快建设以先进制造业为骨干的现代化产业体系，推动工业经济持续平稳向好。
    The training of DeepSeek-V3 is supported by the HAI-LLM framework, an efficient and
    lightweight training framework crafted by our engineers from the ground up. On the whole,
    DeepSeek-V3 applies 16-way Pipeline Parallelism (PP) (Qi et al., 2023a), 64-way Expert Parallelism (EP) (Lepikhin et al., 2021) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP) (Rajbhandari et al., 2020).
    In order to facilitate efficient training of DeepSeek-V3, we implement meticulous engineering
    optimizations. Firstly, we design the DualPipe algorithm for efficient pipeline parallelism.
    Compared with existing PP methods, DualPipe has fewer pipeline bubbles. More importantly, it
    overlaps the computation and communication phases across forward and backward processes,
    thereby addressing the challenge of heavy communication overhead introduced by cross-node
    expert parallelism. Secondly, we develop
    to fully utilize IB and NVLink bandwidths and conserve SMs dedicated to communication. Finally, we meticulously optimize the memory footprint during
    training, thereby enabling us to train DeepSeek-V3 without using costly Tensor Parallelism (TP).
    新华社北京1月17日电（记者邹多为）我国货物贸易第一大国地位更加稳固。国家统计局局长康义17日在国新办发布会上说，2024年，面对世界百年未有之大变局加速演进，外部不稳定不确定因素增多，我国坚定不移推进高水平对外开放，货物贸易规模位居世界第一，为全球发展做出新的重要贡献。
    据海关统计，从总量看，2024年我国货物贸易连续跨过42、43两个万亿级大关，全年进出口总值达到43.85万亿元，同比增长5%，规模再创历史新高，质升量稳目标顺利实现。新一轮促进外贸稳定增长的若干举措相继出台，有力推动四季度外贸实现11.51万亿元的季度历史新高。尤其是12月份，当月进出口规模首次突破4万亿元，增速提升至6.8%，全年外贸圆满收官。
    进出口分开看，2024年，我国出口规模首次突破25万亿元，达到25.45万亿元，同比增长7.1%，连续8年保持增长；同期，我国进口18.39万亿元，同比增长2.3%，连续多年保持世界第二大进口市场。值得一提的是，由于元旦、春节临近，2024年12月份中国消费品进口额创近21个月新高。
    从增量看，2024年我国外贸增长规模达到2.1万亿元，相当于一个中等国家一年的外贸总量。
    从质量看，2024年，我国高技术产品增势良好，电动汽车、3D打印机、工业机器人出口分别实现13.1%、32.8%、45.2%的增长。同期，自主品牌出口创历史新高，跨境电商等新型贸易业态蓬勃发展。
    海关总署副署长王令浚表示，当前，外部环境更加复杂，地缘政治、单边主义和保护主义影响上升，外贸稳增长面临严峻挑战。同时，主要国际经济组织预测今年全球货物贸易将保持增长，我国经济基础稳、优势多、韧性强、潜力大的基本面没有变，推动外贸高质量发展的支撑因素依然稳固。
    """
    query = f"{text}\n以上是一篇文章的句子结构化结果（已经对应了各个句子在文章中的索引），我有一个问题是：2023年中国的gdp是多少？你是一个检索器，请你帮我检索出文章中所有与这个问题语义相近或者直接相关的句子的索引。\n输出格式为List[tuple(start, end)]，例如：[(0, 20), (40, 80), ...]；仅输出结果，不要输出其他解释。"

    a = time.time()
    chunks = TextChunker().chunk(text, chunk_size=1000, overlap=200, return_sentences=True)
    for c in chunks:
        print("======================================================")
        print(c)
    b = time.time()
    print(b-a)
    print(TextChunker().add_start_end(chunks))

