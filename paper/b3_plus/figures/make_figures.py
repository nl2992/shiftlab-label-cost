"""Rebuild the primary result figures for every configured real dataset."""
from make_all_streams import draw, load

if __name__ == "__main__":
    draw(load())
