
import sys

def bayes_theorem(P_D, P_pos_given_D, P_pos_given_not_D):
    # Binary Bayes Rule Example
    P_D = 0.55
    P_not_D = 1 - P_D
    P_pos_given_D = 0.90
    P_pos_given_not_D = 0.15

    # Bayes' Rule
    P_D_given_pos = (P_pos_given_D * P_D) / (
        P_pos_given_D * P_D + P_pos_given_not_D * P_not_D
    )
    return P_D_given_pos




if __name__ == "__main__":
    # Check if the correct number of arguments are provide where there 5 arguments are expected
    if len(sys.argv) != 4:
        print("Usage: python binary_bayes_theorem_example.py <P_D> <P_pos_given_D> <P_pos_given_not_D>")
        sys.exit(1)

    P_D = sys.argv[1]
    P_pos_given_D = sys.argv[2]
    P_pos_given_not_D = sys.argv[3]
    P_D_given_pos = bayes_theorem(P_D,  P_pos_given_D, P_pos_given_not_D)
    print(f"P(Disease | Positive Test) = {P_D_given_pos:.4f}")