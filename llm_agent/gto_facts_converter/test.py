from utils import *
from board import Board
from hand import Hand
def test_evaluate_hand():
    # Test case 1: High card
    hand1 = Hand(['Ah', 'Kd'])
    board1 = Board(['5c', '6d', '7h', '8s', '9c'])
    hand_rankings1, draws1 = evaluate_hand(hand1, board1)
    print("Test case 1 - High card:")
    print("Hand Rankings:", hand_rankings1)
    print("Draws:", draws1)

    # Test case 2: One pair
    hand2 = Hand(['Jh', 'Js'])
    board2 = Board(['6h', '7h', '8h', '9d', 'Tc'])
    hand_rankings2, draws2 = evaluate_hand(hand2, board2)
    print("Test case 2 - One pair:")
    print("Hand Rankings:", hand_rankings2)
    print("Draws:", draws2)

    # # Test case 3: Two pair
    # hand3 = Hand(['Qh', 'Qs'])
    # board3 = Board(['Kh', 'Kd', '8h', '9d', 'Tc'])
    # hand_rankings3, draws3 = evaluate_hand(hand3, board3)
    # print("Test case 3 - Two pair:")
    # print("Hand Rankings:", hand_rankings3)
    # print("Draws:", draws3)

    # # Test case 4: Three of a kind
    # hand4 = Hand(['Kh', 'Ks'])
    # board4 = Board(['Kd', '6h', '8h', '9d', 'Tc'])
    # hand_rankings4, draws4 = evaluate_hand(hand4, board4)
    # print("Test case 4 - Three of a kind:")
    # print("Hand Rankings:", hand_rankings4)
    # print("Draws:", draws4)

    # # Test case 5: Straight
    # hand5 = Hand(['Th', 'Jd'])
    # board5 = Board(['Qh', 'Ks', 'Ac', '2d', '3s'])
    # hand_rankings5, draws5 = evaluate_hand(hand5, board5)
    # print("Test case 5 - Straight:")
    # print("Hand Rankings:", hand_rankings5)
    # print("Draws:", draws5)

    # # Test case 6: Flush
    # hand6 = Hand(['Ah', 'Kh'])
    # board6 = Board(['Qh', 'Jh', 'Th', '3d', '9c'])
    # hand_rankings6, draws6 = evaluate_hand(hand6, board6)
    # print("Test case 6 - Flush:")
    # print("Hand Rankings:", hand_rankings6)
    # print("Draws:", draws6)

    # # Test case 7: Full house
    # hand7 = Hand(['Qd', 'Qs'])
    # board7 = Board(['Qh', '7c', '7d', '2s', '9c'])
    # hand_rankings7, draws7 = evaluate_hand(hand7, board7)
    # print("Test case 7 - Full house:")
    # print("Hand Rankings:", hand_rankings7)
    # print("Draws:", draws7)

    # # Test case 8: Four of a kind
    # hand8 = Hand(['Kd', 'Ks'])
    # board8 = Board(['Kh', 'Kc', '7h', '2s', '9c'])
    # hand_rankings8, draws8 = evaluate_hand(hand8, board8)
    # print("Test case 8 - Four of a kind:")
    # print("Hand Rankings:", hand_rankings8)
    # print("Draws:", draws8)

    # # Test case 9: Straight flush
    # hand9 = Hand(['9h', 'Th'])
    # board9 = Board(['Jh', 'Qh', 'Kh', '2d', '3s'])
    # hand_rankings9, draws9 = evaluate_hand(hand9, board9)
    # print("Test case 9 - Straight flush:")
    # print("Hand Rankings:", hand_rankings9)
    # print("Draws:", draws9)

    # # Test case 10: Royal flush
    # hand10 = Hand(['Th', 'Jh'])
    # board10 = Board(['Ah', 'Kh', 'Qh', '2d', '3s'])
    # hand_rankings10, draws10 = evaluate_hand(hand10, board10)
    # print("Test case 10 - Royal flush:")
    # print("Hand Rankings:", hand_rankings10)
    # print("Draws:", draws10)
    
    # Test case 11: Straight draw
    hand11 = Hand(['9h', 'Th'])
    board11 = Board(['Jh', '5d', '2c'])
    hand_rankings11, draws11 = evaluate_hand(hand11, board11)
    print("Test case 11 - Straight draw:")
    print("Hand Rankings:", hand_rankings11)
    print("Draws:", draws11)


    # Test case 3: Royal flush
    hand3 = Hand(['Th', 'Jh'])
    board3 = Board(['Ah', 'Kh', 'Qh', '2d', '3s'])
    hand_rankings3, draws3 = evaluate_hand(hand3, board3)
    print("Test case 3 - Royal flush:")
    print("Hand Rankings:", hand_rankings3)
    print("Draws:", draws3)


def test_evaluate_hand_with_board_filter():
    # Test case: Overlapping straight draw
    hand = Hand(['9h', '5h'])
    board = Board(['2c', '3c', '4c'])
    hand_rankings, filtered_draws = evaluate_hand_with_board_filter(hand, board)
    print("Test case - Overlapping straight draw:")
    print("Hand Rankings:", hand_rankings)
    print("Filtered Draws:", filtered_draws)

if __name__ == "__main__":
    test_evaluate_hand()
    test_evaluate_hand_with_board_filter()