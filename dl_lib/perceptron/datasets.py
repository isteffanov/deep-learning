
def create_dataset(n: int) -> list[tuple[int, int]]:
    return list(map(lambda x: (x, 2*x), range(n)))