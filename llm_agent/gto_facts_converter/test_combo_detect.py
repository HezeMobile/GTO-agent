import pytest
from board import Board
from hand import Hand
from utils import evaluate_hand, get_top_combinations


def test_high_card_straight():
    # 示例 1: 测试一手高牌和公共牌
    hand1 = Hand(["Ah", "Kd"])
    board1 = Board(["5c", "6d", "7h", "8s", "9c"])
    hand_rankings1, draws1 = evaluate_hand(hand1, board1)
    assert hand_rankings1 == {
        "high_card": 14,
        "one_pair": False,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": True,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": False,
    }
    assert draws1 == {"straight_draw": (False, None), "flush_draw": False, "straight_flush_draw": (False, [])}


def test_one_pair_straight_flush():
    # 示例 2: 测试一手对子和公共牌
    hand2 = Hand(["2h", "2s"])
    board2 = Board(["6h", "7h", "8h", "9d"])
    hand_rankings2, draws2 = evaluate_hand(hand2, board2)
    assert hand_rankings2 == {
        "high_card": 9,
        "one_pair": True,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": False,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": False,
    }
    assert draws2 == {"straight_draw": (True, [[5], [10]]), "flush_draw": (True, 1), "striaght_flush_draw": (False, [])}


def test_royal_flush():
    # 示例 3: 测试同花顺
    hand3 = Hand(["Th", "Jh"])
    board3 = Board(["Ah", "Kh", "Qh", "2d", "3s"])
    hand_rankings3, draws3 = evaluate_hand(hand3, board3)
    assert hand_rankings3 == {
        "high_card": 14,
        "one_pair": False,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": True,
        "flush": True,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": True,
    }
    assert draws3 == {"straight_draw": (False, None), "flush_draw": False, "striaght_flush_draw": (False, [])}


def test_four_of_a_kind():
    # 示例 4: 测试四条
    hand4 = Hand(["8d", "8s"])
    board4 = Board(["8h", "8c", "7h", "2s", "9c"])
    hand_rankings4, draws4 = evaluate_hand(hand4, board4)
    assert hand_rankings4 == {
        "high_card": 9,
        "one_pair": False,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": False,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": True,
        "straight_flush": False,
    }
    assert draws4 == {"straight_draw": (False, None), "flush_draw": False, "striaght_flush_draw": (False, [])}


def test_flush_draw():
    # 示例 5: 测试三张公共牌的情况（翻牌）
    hand5 = Hand(["Kc", "Qc"])
    board5 = Board(["Ac", "5h", "7c"])
    hand_rankings5, draws5 = evaluate_hand(hand5, board5)
    assert hand_rankings5 == {
        "high_card": 14,
        "one_pair": False,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": False,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": False,
    }
    assert draws5 == {
        "straight_draw": (True, [[10, 11]]),
        "flush_draw": (True, 1),
        "striaght_flush_draw": (True, [[(10, "c"), (11, "c")]]),
    }


def test_flush_draw_2():
    # 示例 5: 测试三张公共牌的情况（翻牌）
    hand5 = Hand(["Kc", "Qc"])
    board5 = Board(["Ac", "5h", "7h"])
    hand_rankings5, draws5 = evaluate_hand(hand5, board5)
    assert hand_rankings5 == {
        "high_card": 14,
        "one_pair": False,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": False,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": False,
    }
    assert draws5 == {
        "straight_draw": (True, [[10, 11]]),
        "flush_draw": (True, 2),
        "striaght_flush_draw": (True, [[(10, "c"), (11, "c")]]),
    }


def test_straight_flush_draw():
    # 示例 6: 测试四张公共牌的情况（转牌）
    hand6 = Hand(["9h", "Th"])
    board6 = Board(["Jh", "Qh", "2c", "3d"])
    hand_rankings6, draws6 = evaluate_hand(hand6, board6)
    assert hand_rankings6 == {
        "high_card": 12,
        "one_pair": False,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": False,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": False,
    }
    assert draws6 == {
        "straight_draw": (True, [[8], [13]]),
        "flush_draw": (True, 1),
        "striaght_flush_draw": (True, [[(8, "h")], [(13, "h")]]),
    }


def test_straight_draw():
    hand6 = Hand(["3h", "4h"])
    board6 = Board(["5h", "6h", "3c"])
    hand_rankings6, draws6 = evaluate_hand(hand6, board6)
    assert hand_rankings6 == {
        "high_card": 6,
        "one_pair": True,
        "two_pair": False,
        "three_of_a_kind": False,
        "straight": False,
        "flush": False,
        "full_house": None,
        "four_of_a_kind": False,
        "straight_flush": False,
    }
    assert draws6 == {
        "straight_draw": (True, [[2], [7], [7, 8]]),
        "flush_draw": (True, 1),
        "striaght_flush_draw": (True, [[(2, "h")], [(7, "h")], [(7, "h"), (8, "h")]]),
    }


def test_get_top_combinations_basic():
    hands = ["5c4c", "Ac4c", "5d4d", "Ad4d", "5h4h", "Ah4h"]
    weights = [0.0, 0.0, 2.6768225e-7, 0.34322658, 0.0, 0.0]
    ev = [6.629074, 8.976952, 36.724407, 49.417976, 7.6425858, 9.297714]
    effective_stack = 400
    top_n = 2

    result = get_top_combinations(hands, weights, ev, effective_stack, top_n)

    expected_hands = ["Ad4d", "5d4d"]
    assert result["hands"].tolist() == expected_hands
    assert result["weights"].tolist() == [0.34322658, 2.6768225e-7]
    assert result["ev"].tolist() == [49.417976, 36.724407]


def test_get_top_combinations_all_zero_weights():
    hands = ["5c4c", "Ac4c", "5d4d"]
    weights = [0.0, 0.0, 0.0]
    ev = [6.0, 7.0, 8.0]
    effective_stack = 400
    top_n = 1

    result = get_top_combinations(hands, weights, ev, effective_stack, top_n)

    expected_hand = ["5d4d"]
    assert result["hands"].tolist() == expected_hand
    assert result["weights"].tolist() == [0.0]
    assert result["ev"].tolist() == [8.0]


def test_get_top_combinations_single_combination():
    hands = ["5c4c"]
    weights = [0.2]
    ev = [10.0]
    effective_stack = 400
    top_n = 1

    result = get_top_combinations(hands, weights, ev, effective_stack, top_n)

    expected_hand = ["5c4c"]
    assert result["hands"].tolist() == expected_hand
    assert result["weights"].tolist() == [0.2]
    assert result["ev"].tolist() == [10.0]


def test_get_top_combinations_invalid_top_n():
    hands = ["5c4c", "Ac4c"]
    weights = [0.1, 0.2]
    ev = [5.0, 6.0]
    effective_stack = 400
    top_n = 3  # 大于hands的长度

    result = get_top_combinations(hands, weights, ev, effective_stack, top_n)

    assert len(result) == len(hands)


def test_get_top_combinations_no_ev():
    hands = ["5c4c", "Ac4c"]
    weights = [0.1, 0.2]
    ev = [0.0, 0.0]
    effective_stack = 400
    top_n = 1

    result = get_top_combinations(hands, weights, ev, effective_stack, top_n)

    expected_hand = ["Ac4c"]
    assert result["hands"].tolist() == expected_hand
    assert result["weights"].tolist() == [0.2]
    assert result["ev"].tolist() == [0.0]


def test_get_top_combinations_opp_hands_data():
    hands = [
        "5c4c",
        "Ac4c",
        "5d4d",
        "Ad4d",
        "5h4h",
        "Ah4h",
        "5s4s",
        "As4s",
        "6c5c",
        "7c5c",
        "8c5c",
        "Ac5c",
        "6d5d",
        "7d5d",
        "8d5d",
        "Ad5d",
        "7h5h",
        "8h5h",
        "Ah5h",
        "6s5s",
        "7s5s",
        "8s5s",
        "As5s",
        "6d6c",
        "6s6c",
        "7c6c",
        "8c6c",
        "9c6c",
        "6s6d",
        "7d6d",
        "8d6d",
        "7s6s",
        "8s6s",
        "9s6s",
        "7d7c",
        "7h7c",
        "7s7c",
        "8c7c",
        "9c7c",
        "7h7d",
        "7s7d",
        "8d7d",
        "7s7h",
        "8h7h",
        "9h7h",
        "8s7s",
        "9s7s",
        "8d8c",
        "8h8c",
        "8s8c",
        "9c8c",
        "Ac8c",
        "8h8d",
        "8s8d",
        "Ad8d",
        "8s8h",
        "9h8h",
        "Ah8h",
        "9s8s",
        "As8s",
        "9h9c",
        "9s9c",
        "Kc9c",
        "Ac9c",
        "9s9h",
        "Kh9h",
        "Ah9h",
        "Ks9s",
        "As9s",
        "ThTc",
        "TsTc",
        "JcTc",
        "KcTc",
        "AcTc",
        "TsTh",
        "JhTh",
        "QhTh",
        "KhTh",
        "AhTh",
        "JsTs",
        "QsTs",
        "KsTs",
        "AsTs",
        "JdJc",
        "JhJc",
        "JsJc",
        "KcJc",
        "AcJc",
        "AdJc",
        "AhJc",
        "AsJc",
        "JhJd",
        "JsJd",
        "QdJd",
        "KdJd",
        "AcJd",
        "AdJd",
        "AhJd",
        "AsJd",
        "JsJh",
        "QhJh",
        "KhJh",
        "AcJh",
        "AdJh",
        "AhJh",
        "AsJh",
        "QsJs",
        "KsJs",
        "AcJs",
        "AdJs",
        "AhJs",
        "AsJs",
        "QhQd",
        "QsQd",
        "KcQd",
        "KdQd",
        "KhQd",
        "KsQd",
        "AcQd",
        "AdQd",
        "AhQd",
        "AsQd",
        "QsQh",
        "KcQh",
        "KdQh",
        "KhQh",
        "KsQh",
        "AcQh",
        "AdQh",
        "AhQh",
        "AsQh",
        "KcQs",
        "KdQs",
        "KhQs",
        "KsQs",
        "AcQs",
        "AdQs",
        "AhQs",
        "AsQs",
        "KdKc",
        "KhKc",
        "KsKc",
        "AcKc",
        "AdKc",
        "AhKc",
        "AsKc",
        "KhKd",
        "KsKd",
        "AcKd",
        "AdKd",
        "AhKd",
        "AsKd",
        "KsKh",
        "AcKh",
        "AdKh",
        "AhKh",
        "AsKh",
        "AcKs",
        "AdKs",
        "AhKs",
        "AsKs",
        "AdAc",
        "AhAc",
        "AsAc",
        "AhAd",
        "AsAd",
        "AsAh",
    ]

    weights = [
        0.0,
        0.0,
        2.6768225e-7,
        0.34322658,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.799958e-7,
        0.0010804585,
        0.0,
        0.0,
        0.32922614,
        0.28639603,
        0.42039868,
        1.3482001e-7,
        0.0065211067,
        0.0,
        0.0,
        1.67378e-7,
        0.003012491,
        0.0,
        0.071861975,
        0.15156303,
        0.0,
        0.069121845,
        5.769561e-9,
        0.07899759,
        0.0,
        0.0,
        0.0,
        0.06488876,
        9.967412e-5,
        0.0,
        0.0,
        0.0,
        1.3387734e-5,
        0.0,
        0.0,
        0.0,
        0.40154615,
        0.0,
        0.0013859302,
        0.0,
        4.0386036e-5,
        2.0529748e-7,
        4.0837436e-5,
        0.0,
        0.0,
        0.0030712474,
        0.0,
        0.0024652362,
        5.412266e-5,
        0.21767661,
        0.0,
        0.0489847,
        2.3504779e-7,
        0.02829818,
        1.5700196e-7,
        0.05595135,
        0.03103098,
        0.0015410058,
        0.0,
        0.022941211,
        0.00056555716,
        0.0,
        0.0038694553,
        0.0,
        0.16262491,
        0.16255577,
        0.045307018,
        0.0021862674,
        0.0,
        0.05145374,
        0.08025658,
        0.17331569,
        0.06622863,
        0.0,
        0.08661338,
        0.17117861,
        0.026088936,
        0.0,
        0.04366756,
        0.029926164,
        0.03561609,
        0.23080745,
        0.051954746,
        0.15535395,
        0.04931959,
        0.053434104,
        0.034399603,
        0.0369606,
        0.0,
        0.24940097,
        0.12133663,
        0.21985193,
        0.12067999,
        0.12204805,
        0.04777022,
        0.01710607,
        0.217717,
        0.05500607,
        0.08589807,
        0.047803286,
        0.058244817,
        0.01585296,
        0.20608963,
        0.053222768,
        0.0828029,
        0.045785367,
        0.05660433,
        0.03159258,
        0.03833985,
        6.1982564e-6,
        0.0,
        0.0,
        0.0,
        0.010278985,
        0.02468421,
        0.0074595544,
        0.010079215,
        0.017371695,
        0.012946884,
        0.0004232559,
        0.0022726043,
        0.002775753,
        2.362043e-6,
        0.00010031719,
        0.0,
        6.872217e-6,
        0.011503295,
        0.00047513327,
        0.001764687,
        0.0025216579,
        1.9442807e-6,
        9.80793e-5,
        0.0,
        3.1154525e-6,
        0.24001831,
        0.2773977,
        0.22424503,
        0.035021894,
        0.12578237,
        0.06601465,
        0.063628554,
        0.28655463,
        0.2442618,
        0.19393273,
        0.022512432,
        0.20605178,
        0.19190857,
        0.27813697,
        0.047747366,
        0.12108436,
        0.02308622,
        0.04758573,
        0.05832091,
        0.11665046,
        0.061076604,
        0.030348781,
        0.06499093,
        0.007970602,
        0.008430728,
        0.06528828,
        0.06630619,
        0.006984226,
    ]

    ev = [
        6.629074,
        8.976952,
        36.724407,
        49.417976,
        7.6425858,
        9.297714,
        6.6223373,
        8.957443,
        35.2434,
        16.654425,
        26.042082,
        9.713364,
        90.317696,
        47.549038,
        68.143745,
        46.822815,
        17.735626,
        27.278738,
        9.960945,
        35.255623,
        16.654402,
        26.245922,
        9.697266,
        202.71556,
        198.20277,
        40.44326,
        49.333633,
        130.69957,
        202.71185,
        104.381325,
        123.94962,
        40.465816,
        49.629803,
        132.11148,
        41.638306,
        41.22283,
        40.460625,
        232.53033,
        53.128696,
        42.40249,
        41.652927,
        276.85965,
        41.254955,
        232.02644,
        54.171127,
        231.29846,
        53.657406,
        55.62761,
        55.162014,
        55.139877,
        62.352795,
        28.226723,
        56.678337,
        56.075394,
        85.260086,
        55.51067,
        63.602272,
        28.417336,
        63.28544,
        28.35678,
        215.75775,
        214.40372,
        62.796143,
        54.409344,
        214.86522,
        63.741127,
        55.227562,
        63.45134,
        54.711796,
        231.60835,
        231.60986,
        93.56227,
        73.68849,
        67.981735,
        230.081,
        94.59469,
        155.69504,
        75.108475,
        69.333115,
        94.62275,
        155.77122,
        74.5097,
        68.698784,
        117.64807,
        115.04723,
        115.059105,
        361.37827,
        35.733387,
        41.11858,
        35.90311,
        35.699745,
        117.920525,
        117.92769,
        186.22974,
        391.8136,
        37.003567,
        113.8556,
        37.19524,
        36.97485,
        115.319336,
        121.22821,
        360.7956,
        36.037136,
        41.64808,
        36.205948,
        35.998203,
        121.20526,
        360.80276,
        36.070908,
        41.680187,
        36.243572,
        36.030743,
        251.24144,
        251.2844,
        112.190926,
        179.22632,
        112.89108,
        112.26397,
        98.40815,
        175.07701,
        99.08047,
        98.40863,
        246.61572,
        110.4694,
        115.237076,
        111.093605,
        110.54545,
        96.63107,
        103.196495,
        97.3547,
        96.63377,
        110.46563,
        115.23058,
        111.111595,
        110.52549,
        96.663895,
        103.23845,
        97.39206,
        96.659676,
        139.0384,
        134.49742,
        133.95284,
        25.535845,
        30.747135,
        25.9462,
        25.876322,
        139.60892,
        139.0246,
        26.914333,
        102.00645,
        27.04281,
        26.920242,
        134.47354,
        25.919823,
        30.754787,
        25.643253,
        25.922897,
        25.879303,
        30.689247,
        25.945831,
        25.557121,
        102.3387,
        96.86234,
        96.307304,
        102.98457,
        102.34532,
        96.89494,
    ]

    effective_stack = 400
    top_n = 5

    result = get_top_combinations(hands, weights, ev, effective_stack, top_n)

    print(result)

    # expected_hands = [...]
    # assert result['hands'].tolist() == expected_hands


# def test_get_top_combinations_in_hands():
#     hands = ["2d2c", "2h2c", "2s2c", "Ac2c", "2h2d", "2s2d", "Ad2d", "2s2h", "Ah2h", "As2s", "3d3c", "3h3c",
#              "3s3c", "5c3c", "Ac3c", "3h3d", "3s3d", "5d3d", "Ad3d", "3s3h", "5h3h", "Ah3h", "5s3s", "As3s",
#              "4d4c", "4h4c", "4s4c", "5c4c", "6c4c", "Ac4c", "4h4d", "4s4d", "5d4d", "6d4d", "Ad4d", "4s4h",
#              "5h4h", "Ah4h", "5s4s", "6s4s", "As4s", "5d5c", "5h5c", "5s5c", "6c5c", "7c5c", "Kc5c", "Ac5c",
#              "5h5d", "5s5d", "6d5d", "7d5d", "Kd5d", "Ad5d", "5s5h", "7h5h", "Kh5h", "Ah5h", "6s5s", "7s5s",
#              "Ks5s", "As5s", "6d6c", "6s6c", "7c6c", "8c6c", "9c6c", "Kc6c", "Ac6c", "6s6d", "7d6d", "8d6d",
#              "Kd6d", "Ad6d", "7s6s", "8s6s", "9s6s", "Ks6s", "As6s", "7d7c", "7h7c", "7s7c", "8c7c", "9c7c",
#              "Tc7c", "Kc7c", "Ac7c", "7h7d", "7s7d", "8d7d", "Kd7d", "Ad7d", "7s7h", "8h7h", "9h7h", "Th7h",
#              "Kh7h", "Ah7h", "8s7s", "9s7s", "Ts7s", "Ks7s", "As7s", "8d8c", "8h8c", "8s8c", "9c8c", "Tc8c",
#              "Jc8c", "Kc8c", "Ac8c", "8h8d", "8s8d", "Jd8d", "Qd8d", "Kd8d", "Ad8d", "8s8h", "9h8h", "Th8h",
#              "Jh8h", "Qh8h", "Kh8h", "Ah8h", "9s8s", "Ts8s", "Js8s", "Qs8s", "Ks8s", "As8s", "9h9c", "9s9c",
#              "Tc9c", "Jc9c", "Kc9c", "Ac9c", "9s9h", "Th9h", "Jh9h", "Qh9h", "AhTs", "AsTs", "JdJc", "JhJc",
#              "JsJc", "KcJc", "KdJc", "KhJc", "KsJc", "AcJc", "AdJc", "AhJc", "AsJc", "JhJd", "JsJd", "QdJd",
#              "KcJd", "KdJd", "KhJd", "KsJd", "AcJd", "AdJd", "AhJd", "AsJd", "JsJh", "QhJh", "KcJh", "KdJh",
#              "KhJh", "KsJh", "AcJh", "AdJh", "AhJh", "AsJh", "QsJs", "KcJs", "KdJs", "KhJs", "KsJs", "AcJs",
#              "AdJs", "AhJs", "AsJs", "QhQd", "QsQd", "KcQd", "KdQd", "KhQd", "KsQd", "AcQd", "AdQd", "AhQd",
#              "AsQd", "QsQh", "KcQh", "KdQh", "KhQh", "KsQh", "AcQh", "AdQh", "AhQh", "AsQh", "KcQs", "KdQs",
#              "KhQs", "KsQs", "AcQs", "AdQs", "AhQs", "AsQs", "AdKc", "AhKc", "AsKc", "AcKd", "AhKd", "AsKd",
#              "AcKh", "AdKh", "AsKh", "AcKs", "AdKs", "AhKs"]
#
#     weights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.999165, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6957108,
#                0.9985092, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.5813393e-6, 0.0, 0.0, 0.0, 0.99987, 0.9999922,
#                0.9995524, 0.0, 0.0, 0.0, 0.0, 6.326654e-6, 0.0, 0.0, 0.0, 0.0, 3.3015433e-5, 0.0, 0.0, 0.0, 0.0,
#                1.0, 0.9991916, 0.99989736, 0.99729824, 0.0, 0.0, 0.0, 0.0, 4.8945563e-5, 0.0, 0.0, 0.0, 0.9733859,
#                0.90482384, 0.30829075, 0.9896652, 1.0, 0.7243935, 1.0938631e-5, 0.9753287, 0.017063294, 0.6177557,
#                0.91994184, 1.0, 0.31486323, 0.9954857, 1.0, 0.7289058, 1.6869775e-5, 0.18048769, 1.2579634e-6,
#                1.5237321e-6, 0.11101371, 0.52139604, 0.51233876, 0.00065826176, 0.0, 0.17462198, 0.18755546,
#                0.19654112, 0.6134594, 0.8241679, 1.4164815e-6, 0.18845162, 0.5380286, 0.45015782, 0.002232625, 0.0,
#                0.105233826, 0.5418202, 0.45432898, 0.00023543251, 0.0, 0.48932797, 0.9786475, 0.99754757, 0.99688756,
#                0.9935729, 0.012097698, 0.12716678, 0.0, 0.44581258, 0.4983057, 0.40787798, 0.11767847, 0.50966334,
#                0.49988046, 0.9896601, 0.9994313, 0.98785716, 0.08517838, 0.72442394, 0.10889387, 0.0, 0.9994543,
#                0.9905372, 0.010507059, 0.7270972, 0.14192505, 0.0, 0.97767526, 0.9554198, 0.99998504, 0.9999267,
#                0.99911827, 0.014204624, 0.94077265, 1.0, 0.9999013, 0.96644384, 0.9987604, 0.0013396749, 0.9999999,
#                0.99993783, 0.9685714, 0.9995256, 0.0038424432, 0.59498346, 0.6056489, 0.993594, 0.8137131, 0.35432655,
#                0.00010519249, 0.31437463, 0.35661972, 0.7050343, 0.98468053, 0.7333337, 0.86940765, 0.38533264, 0.0,
#                0.34344396, 0.38791904, 0.96493614, 0.7503975, 0.8784235, 0.38087898, 0.0, 0.339756, 0.38368386, 1.0,
#                0.8629887, 0.858502, 0.25326523, 0.23616864, 0.22820528, 0.25660148, 0.78947055, 0.4224209, 0.8022541,
#                0.79190385, 0.99944395, 0.99968904, 0.06692069, 0.21441366, 0.07878328, 0.19456328, 0.22403702,
#                0.31692937, 0.7025065, 0.35739005, 0.31799495, 0.84223044, 0.98989886, 0.27440643, 0.24556038,
#                0.24975725, 0.27774024, 0.77625424, 0.37205592, 0.7936409, 0.77740324, 0.9913359, 0.2860893, 0.25781652,
#                0.26172364, 0.2897846, 0.77684104, 0.3695354, 0.7938275, 0.7781669, 0.17107041, 0.16972399, 0.7571274,
#                0.0126098925, 0.7189271, 0.7655807, 0.96526676, 1.0, 0.9755187, 0.9708119, 0.25321814, 0.7498761,
#                0.9740098, 0.702908, 0.74700475, 0.9115402, 0.23187704, 0.8967464, 0.9180749, 0.7463671, 0.96728116,
#                0.70042944, 0.7428552, 0.90695417, 0.23153089, 0.8920884, 0.91363406, 0.05455149, 0.92439336,
#                0.97072345, 0.8137739, 0.8203084, 0.8159263, 0.9600236, 0.073783144, 0.96141964, 0.96569574,
#                0.055904374, 0.9188667]
#
#     ev = [28.743866, 27.398224, 27.423126, 11.418579, 28.74334, 28.7761, 83.18661, 27.423302, 11.679779, 11.416634,
#           28.725143, 27.406525, 27.417786, 11.448944, 11.336266, 28.739815, 28.752045, 64.948364, 83.011475, 27.429726,
#           12.666077, 11.596359, 11.452301, 11.330887, 27.802513, 26.65606, 26.387955, 10.459213, 43.082302, 11.48793,
#           28.074944, 27.802895, 66.15735, 101.95984, 82.82409, 26.65541, 11.609932, 11.77739, 10.476807, 43.132534,
#           11.48967, 28.689903, 27.640495, 27.155586, 42.923203, 22.219643, 32.381836, 10.554665, 29.382103, 28.796082,
#           103.40782, 77.954865, 93.842735, 83.50014, 27.720802, 24.0915, 33.656624, 11.73407, 43.035023, 22.325562,
#           32.3553, 10.620918, 227.07434, 224.93631, 53.617977, 69.0919, 173.51933, 61.009933, 40.864193, 227.10922,
#           115.22611, 132.25888, 124.69178, 112.16993, 53.687378, 69.341934, 172.93478, 61.125504, 40.866127, 48.05379,
#           47.46221, 46.57171, 305.98126, 67.74441, 76.057755, 46.412796, 23.770683, 48.935272, 48.03293, 326.95978,
#           107.619774, 97.30574, 47.45294, 303.9028, 68.40542, 77.73089, 47.66278, 24.124176, 305.87268, 67.65065,
#           76.97195, 46.411514, 23.911865, 72.05124, 70.4515, 69.70034, 82.98688, 89.748055, 336.8401, 57.56518,
#           44.460045, 72.96526, 72.13492, 353.97363, 168.82935, 119.48188, 119.83954, 70.53032, 83.67469, 90.9084,
#           334.01343, 114.385635, 59.54605, 46.182507, 82.962524, 90.31886, 335.82556, 113.95307, 57.82344, 44.719376,
#           251.41267, 249.0056, 177.34302, 93.054634, 79.235214, 63.339855, 251.16963, 180.24997, 93.38749, 185.97835,
#           79.55678, 63.605984, 178.19522, 93.362015, 183.94391, 79.18309, 63.3027, 275.92712, 276.0719, 100.329704,
#           88.88074, 71.32799, 75.9176, 71.349304, 71.18127, 273.18402, 101.54747, 193.08395, 90.07938, 72.15402,
#           76.847984, 72.54733, 72.18499, 101.50033, 193.16684, 89.75544, 72.06813, 76.7542, 72.26664, 72.293915,
#           109.771736, 107.12701, 107.08821, 356.29684, 362.3192, 357.68176, 358.08392, 61.517475, 70.15576, 61.67819,
#           61.540115, 110.73762, 110.69881, 180.93246, 361.17984, 380.94778, 360.76712, 361.19177, 64.31621, 136.20343,
#           64.24504, 64.32802, 107.8796, 122.24993, 356.45233, 360.75797, 354.28302, 356.46817, 62.00158, 70.1547,
#           62.121834, 62.02975, 122.29268, 356.53125, 360.83853, 356.1365, 354.77026, 61.953617, 70.11636, 62.075153,
#           61.9815, 296.2114, 296.23138, 126.24687, 183.47119, 126.07048, 126.15627, 124.16424, 178.57846, 124.27555,
#           124.16369, 291.88995, 124.504974, 128.49942, 124.349396, 124.394264, 122.868546, 127.470535, 123.08577,
#           122.87144, 124.511696, 128.50508, 124.384094, 124.387695, 122.89723, 127.47781, 123.11017, 122.90013,
#           54.5797, 49.090332, 48.865723, 51.684574, 51.639927, 51.74925, 48.898235, 54.618267, 48.923126, 48.803715,
#           54.575375, 49.04845]
#
#     effective_stack = 400
#     top_n = 5
#
#     result = get_top_combinations(hands, weights, ev, effective_stack, top_n)
#
#     print(result)
#
#     # expected_hands = [...]
#     # assert result['hands'].tolist() == expected_hands


if __name__ == "__main__":
    pytest.main()
