# %%

import gc
import time
import numpy as np
import numpy.typing as npt
import os
import logging
import pickle
import matplotlib.pyplot as plt


from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from datetime import timedelta
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable


# %%


@dataclass(frozen=True)
class MonteCarloResult:

    results: npt.NDArray[np.int_]
    lattice_size: float
    temp: float
    beta: float
    j_coupl: float
    kB: float
    eq_steps: int
    sim_steps: int
    flip_frac: float

    def show_lattice(
        self,
        step: int,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 300,
    ) -> Tuple[plt.Figure, plt.Axes, plt.Colorbar]:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(self.results[step], cmap=cmap, vmin=-1, vmax=1, origin="lower")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{5}%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical")

        return fig, ax, cbar

    def animate_lattice(
        self,
        filename: str,
        fps: int = 200,
        cmap: str = "viridis",
        figsize: Tuple[float, float] = (8, 8),
        dpi: int = 200,  # animation scales badly with dpi
    ) -> None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(self.results[0], cmap=cmap, vmin=-1, vmax=1, origin="lower")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size=f"{5}%", pad=0.2)
        cbar = plt.colorbar(im, cax=cax, orientation="vertical")
        title = ax.set_title(f"Step 0")

        def update(step):
            im.set_array(self.results[step + 1])
            title.set_text(f"Step {step + 1}")

            return im, title

        anim = FuncAnimation(fig, update, frames=len(self.results) - 1, blit=True)

        anim.save(
            f"{filename}.mp4",
            writer="ffmpeg",  # may need to install this and tell matplotlib where the writer path is
            fps=fps,
        )


def calc_energy(lattice: np.ndarray, J: float = 1.0) -> float:
    # Sum nearest-neighbor interactions (using PBCs)
    energy = -J * np.sum(
        lattice
        * (  # ngl ChatGPT did this and its cracked
            np.roll(lattice, shift=1, axis=0)  # Shift up
            + np.roll(lattice, shift=-1, axis=0)  # Shift down
            + np.roll(lattice, shift=1, axis=1)  # Shift right
            + np.roll(lattice, shift=-1, axis=1)  # Shift left
        )
    )

    return energy / 2  # Each pair counted twice (I think)


def calc_magnetisation(lattice: npt.NDArray[np.int_]) -> np.int_:
    return np.sum(lattice, dtype=np.int_)


def mc_step(
    lattice: npt.NDArray[np.int_],
    beta: float | int,
    j_coupl: float | int = 1.0,
    flip_frac: float = 0.1,  # provided this is small it should be fine
) -> npt.NDArray[np.int_]:

    l_size = lattice.shape[0]

    sel = np.random.rand(l_size, l_size) < flip_frac

    new_lattice = lattice.copy()

    new_lattice[sel] *= -1

    # probably the most expensive part of the simulation
    E_old = calc_energy(lattice, J=j_coupl)
    E_new = calc_energy(new_lattice, J=j_coupl)

    delta_E = E_new - E_old

    if delta_E < 0 or (np.random.rand() < np.exp(-beta * delta_E)):
        return new_lattice  # Accept the new configuration
    else:
        return lattice  # Keep the old configuration


def run_mc(
    eq_steps: int = 3000,
    sim_steps: int = 10000,
    kB: float = 1.0,
    coupl_const: float = 1.0,
    temp: float = 2.6,
    lat_size: int = 10,
    flip_frac: float = 0.1,
) -> MonteCarloResult:

    start = time.time()

    beta = 1 / (kB * temp)

    init_lattice = np.random.choice([-1, 1], size=(lat_size, lat_size))
    curr_lattice = init_lattice

    for i in range(eq_steps):
        curr_lattice = mc_step(
            lattice=curr_lattice, beta=beta, j_coupl=coupl_const, flip_frac=flip_frac
        )

    sim_results = np.full(
        (sim_steps + 1, lat_size, lat_size), fill_value=np.nan, dtype=np.int_
    )

    sim_results[0] = curr_lattice

    for i in range(1, sim_steps + 1):
        sim_results[i] = mc_step(
            lattice=sim_results[i - 1],
            beta=beta,
            j_coupl=coupl_const,
            flip_frac=flip_frac,
        )

    gc.collect()

    end = time.time()
    time_taken = round(end - start)

    print(
        f"Sim complete in {timedelta(seconds=time_taken)} (size={lat_size}, temp={temp})."
    )

    return MonteCarloResult(
        results=sim_results,
        lattice_size=lat_size,
        temp=temp,
        beta=beta,
        j_coupl=coupl_const,
        kB=kB,
        eq_steps=eq_steps,
        sim_steps=sim_steps,
        flip_frac=flip_frac,
    )


def parallel_process(arg_list: List[Dict], function: Callable, n_jobs: int = 1):

    results = []

    with ProcessPoolExecutor(max_workers=n_jobs) as pool:

        futures = [pool.submit(function, **kwargs) for kwargs in arg_list]

        tqdm_kwargs = {
            "total": len(futures),
            "unit": "sims",
            "unit_scale": True,
            "leave": True,
        }

        for f in tqdm(as_completed(futures), **tqdm_kwargs):
            pass

    for i, future in enumerate(futures):
        try:
            results.append(future.result())
        except Exception as e:
            logger.warning(f"Caught exception: {e}")

    return results


def save_to_pickle(data: Any, filepath: str):

    with open(f"{filepath}.pickle", "wb") as f:
        pickle.dump(data, f)


# %%

if __name__ == "__main__":

    logger = logging.getLogger(__name__)

    eq_steps = (
        30000  # probably needs to be much higher, I can't be bothered to investigate
    )
    sim_steps = (
        1000000  # probably needs to be much higher, I can't be bothered to investigate
    )
    kB = 1.0  # change as needed
    coupl_const = 1.0  # change as needed
    flip_frac = 0.1  # change as needed, but keep small

    temperatures = np.linspace(1.6, 3.6, 3)  # change as needed
    lattice_sizes = np.arange(3, 6, 2, dtype=int)  # change as needed

    args = []

    for lat_size in lattice_sizes:

        for temp in temperatures:

            args.append(
                {
                    "eq_steps": eq_steps,
                    "sim_steps": sim_steps,
                    "kB": sim_steps,
                    "coupl_const": coupl_const,
                    "temp": temp,
                    "lat_size": lat_size,
                    "flip_frac": flip_frac,
                }
            )

    # the issue with parallelising it this way is that each lattice always starts "cold"
    # if each worker was given a range of temperatures you could feed the end result of the last temp
    # into the starting state of the next temp, this would save equilibration time
    # or you could just equillibrate for a long time...
    results = parallel_process(
        arg_list=args, function=run_mc, n_jobs=os.cpu_count() - 2 or 1
    )

    save_to_pickle(results, "ising_mc_results.pkl")

# %%
