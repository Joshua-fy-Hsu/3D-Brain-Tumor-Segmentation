"""Plain-language descriptions for AAL3 regions, bilingual (en / zh).

Ported from the frontend ANATOMY_MAP tables (static/app.js) so the PDF report
can show the same friendly "what this region does" text the web UI shows.
Each entry maps a prefix (matched against the hemisphere-stripped AAL3 base
name) to (plain_name, lobe, role) for each language.
"""
from __future__ import annotations

import re

# (regex, en(plain, lobe, role), zh(plain, lobe, role))
_TABLE = [
    (r"^Precentral",        ("Precentral gyrus", "Frontal lobe", "primary motor cortex - voluntary movement"),
                            ("中央前迴", "額葉", "初級運動皮質 - 自主運動")),
    (r"^Postcentral",       ("Postcentral gyrus", "Parietal lobe", "primary somatosensory cortex - touch & body sense"),
                            ("中央後迴", "頂葉", "初級體感皮質 - 觸覺與身體感覺")),
    (r"^Frontal_Sup_Medial", ("Superior medial frontal", "Frontal lobe", "decision-making & social cognition"),
                            ("上內側額葉", "額葉", "決策與社會認知")),
    (r"^Frontal_Sup",       ("Superior frontal gyrus", "Frontal lobe", "working memory & planning"),
                            ("上額迴", "額葉", "工作記憶與規劃")),
    (r"^Frontal_Mid",       ("Middle frontal gyrus", "Frontal lobe", "attention & executive function"),
                            ("中額迴", "額葉", "注意力與執行功能")),
    (r"^Frontal_Inf",       ("Inferior frontal gyrus", "Frontal lobe", "speech production (Broca's area, left)"),
                            ("下額迴", "額葉", "語言產生（布若卡區，左側）")),
    (r"^Frontal_Med_Orb",   ("Medial orbitofrontal", "Frontal lobe", "reward & emotional regulation"),
                            ("內側眶額", "額葉", "獎賞與情緒調節")),
    (r"^OFC|^Frontal.*Orb",  ("Orbitofrontal cortex", "Frontal lobe", "reward, decision-making & impulse control"),
                            ("眶額皮質", "額葉", "獎賞、決策與衝動控制")),
    (r"^Rectus",            ("Gyrus rectus", "Frontal lobe", "olfactory & limbic processing"),
                            ("直迴", "額葉", "嗅覺與邊緣系統處理")),
    (r"^Olfactory",         ("Olfactory cortex", "Frontal lobe", "sense of smell"),
                            ("嗅覺皮質", "額葉", "嗅覺")),
    (r"^Supp_Motor_Area",   ("Supplementary motor area", "Frontal lobe", "movement planning & sequencing"),
                            ("輔助運動區", "額葉", "動作規劃與排序")),
    (r"^Rolandic_Oper",     ("Rolandic operculum", "Frontal lobe", "face/mouth motor control & taste"),
                            ("中央溝蓋部", "額葉", "臉部／口部運動控制與味覺")),
    (r"^Paracentral_Lobule", ("Paracentral lobule", "Frontal/Parietal", "motor & sensory control of the legs"),
                            ("旁中央小葉", "額／頂葉", "下肢的運動與感覺控制")),
    (r"^Insula",            ("Insular cortex", "Insula", "interoception, taste & emotion"),
                            ("腦島皮質", "腦島", "內感受、味覺與情緒")),
    (r"^Cingulate_Ant|^ACC", ("Anterior cingulate", "Limbic system", "emotion, error detection & motivation"),
                            ("前扣帶迴", "邊緣系統", "情緒、錯誤偵測與動機")),
    (r"^Cingulate_Mid",     ("Mid cingulate", "Limbic system", "pain processing & action selection"),
                            ("中扣帶迴", "邊緣系統", "疼痛處理與動作選擇")),
    (r"^Cingulate_Post",    ("Posterior cingulate", "Limbic system", "memory recall & self-reflection"),
                            ("後扣帶迴", "邊緣系統", "記憶回想與自我反思")),
    (r"^Hippocampus",       ("Hippocampus", "Limbic system", "memory formation & spatial navigation"),
                            ("海馬迴", "邊緣系統", "記憶形成與空間導航")),
    (r"^ParaHippocampal",   ("Parahippocampal gyrus", "Limbic system", "memory encoding & scene recognition"),
                            ("海馬旁迴", "邊緣系統", "記憶編碼與場景辨識")),
    (r"^Amygdala",          ("Amygdala", "Limbic system", "fear, emotion & threat detection"),
                            ("杏仁核", "邊緣系統", "恐懼、情緒與威脅偵測")),
    (r"^Calcarine",         ("Calcarine cortex", "Occipital lobe", "primary visual cortex (V1) - core vision"),
                            ("距狀溝皮質", "枕葉", "初級視覺皮質（V1）- 核心視覺")),
    (r"^Cuneus",            ("Cuneus", "Occipital lobe", "basic visual processing"),
                            ("楔葉", "枕葉", "基本視覺處理")),
    (r"^Lingual",           ("Lingual gyrus", "Occipital lobe", "vision - letters, words & faces"),
                            ("舌迴", "枕葉", "視覺 - 字母、文字與臉孔")),
    (r"^Occipital",         ("Occipital gyrus", "Occipital lobe", "higher-order visual processing"),
                            ("枕迴", "枕葉", "高階視覺處理")),
    (r"^Fusiform",          ("Fusiform gyrus", "Temporal/Occipital", "face & object recognition"),
                            ("梭狀迴", "顳／枕葉", "臉孔與物件辨識")),
    (r"^Parietal_Sup",      ("Superior parietal lobule", "Parietal lobe", "spatial orientation & attention"),
                            ("上頂小葉", "頂葉", "空間定向與注意力")),
    (r"^Parietal_Inf",      ("Inferior parietal lobule", "Parietal lobe", "language, math & spatial reasoning"),
                            ("下頂小葉", "頂葉", "語言、數學與空間推理")),
    (r"^SupraMarginal",     ("Supramarginal gyrus", "Parietal lobe", "language perception & phonology"),
                            ("緣上迴", "頂葉", "語言感知與語音處理")),
    (r"^Angular",           ("Angular gyrus", "Parietal lobe", "reading, math & memory retrieval"),
                            ("角迴", "頂葉", "閱讀、數學與記憶提取")),
    (r"^Precuneus",         ("Precuneus", "Parietal lobe", "self-awareness & visuospatial imagery"),
                            ("楔前葉", "頂葉", "自我覺察與視覺空間想像")),
    (r"^Heschl",            ("Heschl's gyrus", "Temporal lobe", "primary auditory cortex - hearing"),
                            ("赫氏迴", "顳葉", "初級聽覺皮質 - 聽覺")),
    (r"^Temporal_Sup",      ("Superior temporal gyrus", "Temporal lobe", "auditory processing & language (Wernicke)"),
                            ("上顳迴", "顳葉", "聽覺處理與語言（韋尼克區）")),
    (r"^Temporal_Mid",      ("Middle temporal gyrus", "Temporal lobe", "word meaning & visual motion"),
                            ("中顳迴", "顳葉", "詞義與視覺動態")),
    (r"^Temporal_Inf",      ("Inferior temporal gyrus", "Temporal lobe", "object & visual recognition"),
                            ("下顳迴", "顳葉", "物件與視覺辨識")),
    (r"^Temporal_Pole",     ("Temporal pole", "Temporal lobe", "social & emotional processing"),
                            ("顳極", "顳葉", "社會與情緒處理")),
    (r"^Caudate",           ("Caudate nucleus", "Basal ganglia", "movement control & learning"),
                            ("尾狀核", "基底核", "運動控制與學習")),
    (r"^Putamen",           ("Putamen", "Basal ganglia", "movement regulation & motor skills"),
                            ("殼核", "基底核", "運動調節與動作技能")),
    (r"^Pallidum",          ("Globus pallidus", "Basal ganglia", "movement inhibition & posture"),
                            ("蒼白球", "基底核", "運動抑制與姿勢")),
    (r"^N_Acc",             ("Nucleus accumbens", "Basal ganglia", "reward, pleasure & motivation"),
                            ("依核", "基底核", "獎賞、愉悅與動機")),
    (r"^Thalamus|^Thal_",    ("Thalamus", "Diencephalon", "sensory & motor relay hub"),
                            ("視丘", "間腦", "感覺與運動的中繼樞紐")),
    (r"^Cerebellum|^Vermis",  ("Cerebellum", "Cerebellum", "balance, coordination & fine movement"),
                            ("小腦", "小腦", "平衡、協調與精細動作")),
    (r"^VTA",               ("Ventral tegmental area", "Midbrain", "dopamine reward pathway"),
                            ("腹側被蓋區", "中腦", "多巴胺獎賞路徑")),
    (r"^SN_",               ("Substantia nigra", "Midbrain", "dopamine production & movement"),
                            ("黑質", "中腦", "多巴胺生成與運動")),
    (r"^Red_N",             ("Red nucleus", "Midbrain", "motor coordination"),
                            ("紅核", "中腦", "運動協調")),
    (r"^LC_",               ("Locus coeruleus", "Brainstem", "arousal, attention & stress response"),
                            ("藍斑核", "腦幹", "覺醒、注意力與壓力反應")),
    (r"^Raphe",             ("Raphe nuclei", "Brainstem", "serotonin regulation & mood"),
                            ("縫核", "腦幹", "血清素調節與情緒")),
]

_COMPILED = [(re.compile(rx), en, zh) for rx, en, zh in _TABLE]


def describe(name: str, lang: str = "en") -> dict:
    """Return {title, lobe, role, code} for an AAL3 label name."""
    hemi_en = ""
    hemi_zh = ""
    if name.endswith("_L"):
        hemi_en, hemi_zh = "Left ", "左 "
    elif name.endswith("_R"):
        hemi_en, hemi_zh = "Right ", "右 "
    base = re.sub(r"_[LR]$", "", name)
    for rx, en, zh in _COMPILED:
        if rx.search(base):
            plain, lobe, role = zh if lang == "zh" else en
            hemi = hemi_zh if lang == "zh" else hemi_en
            return {"title": f"{hemi}{plain}", "lobe": lobe, "role": role, "code": name}
    # fallback
    pretty = base.replace("_", " ")
    if lang == "zh":
        hemi = hemi_zh
        return {"title": f"{hemi}{pretty}", "lobe": "腦區域",
                "role": "AAL3 圖譜標記的腦結構", "code": name}
    return {"title": f"{hemi_en}{pretty}", "lobe": "Brain region",
            "role": "labelled brain structure (AAL3 atlas)", "code": name}
