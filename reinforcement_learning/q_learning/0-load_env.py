import gym


def load_frozen_lake(
    desc: list[list] | None = None,
    map_name: str | None = None,
    is_slippery: bool = False,
) -> gym.Wrapper:
    return gym.make(
        "FrozenLake-v1",
        desc=desc,
        map_name=map_name,
        is_slippery=is_slippery,
        render_mode="ansi",
    )
