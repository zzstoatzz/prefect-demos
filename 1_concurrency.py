# /// script
# dependencies = [
#     "prefect"
# ]
# ///

import httpx
from prefect import flow, task


@task
def fetch_pokemon(number: int) -> dict:
    """Fetch a single pokemon from the API"""
    with httpx.Client() as client:
        response = client.get(f"https://pokeapi.co/api/v2/pokemon/{number}")
        response.raise_for_status()
        return response.json()


@task
def extract_pokemon_data(pokemon: dict) -> dict:
    """Extract relevant data from pokemon response"""
    return {
        "name": pokemon["name"],
        "height": pokemon["height"],
        "weight": pokemon["weight"],
    }


@flow(log_prints=True)
def pokemon_stats(n_pokemon: int):
    """Concurrent pokemon data fetching"""

    raw_pokemon_data = fetch_pokemon.map(range(1, n_pokemon + 1))

    pokemon_stats = extract_pokemon_data.map(raw_pokemon_data).result()

    print("Pokemon stats:")
    for stats in pokemon_stats:
        print(f"\t{stats['name']}: {stats['weight']}kg, {stats['height']}0cm")

    return pokemon_stats


if __name__ == "__main__":
    pokemon_stats(n_pokemon=5)
