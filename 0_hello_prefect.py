# /// script
# dependencies = [
#     "prefect",
# ]
# ///

from prefect import flow


def normal_python():
    """Just a regular Python function doing its thing"""
    result = 2 + 2
    print(f"Regular Python calculation: {result}")
    return result


def calling_some_3rd_party():
    """Simulate any 3rd party library call"""
    print("3rd party library returned some response")
    return "some response"


@flow(log_prints=True)
def business_logic():
    """Your existing code, now with Prefect observability"""
    normal_python()
    calling_some_3rd_party()


if __name__ == "__main__":
    business_logic()
