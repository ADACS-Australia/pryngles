#!/usr/bin/env python
##################################################################
#                                                                #
# .#####...#####...##..##..##..##...####...##......######...####..#
# .##..##..##..##...####...###.##..##......##......##......##.....#
# .#####...#####.....##....##.###..##.###..##......####.....####..#
# .##......##..##....##....##..##..##..##..##......##..........##.#
# .##......##..##....##....##..##...####...######..######...####..#
# ................................................................#
#                                                                #
# PlanetaRY spanGLES                                             #
#                                                                #
##################################################################
# License http://github.com/seap-udea/pryngles-public            #
##################################################################
# coding: utf-8
"""
This code is intended to generate the sampler presets of Pryngles.
"""

from tqdm import tqdm

from pryngles.common import VERB_NONE, Verbose
from pryngles.consts import SAMPLER_CIRCLE_PRESETS, SAMPLER_SPHERE_PRESETS
from pryngles.misc import get_data
from pryngles.sampler import Sampler

Verbose.VERBOSITY = VERB_NONE
nsearch = 2  # Larger than 1

print("Generating the spherical presets...")
for N in tqdm(SAMPLER_SPHERE_PRESETS):
    dran_min = 1e100
    sp_min = None
    for seed in range(1, nsearch):
        sp = Sampler(N=N, seed=seed)
        sp.gen_sphere()
        sp.purge_sample()
        # print(seed,sp.dmin,sp.dmax,sp.dran/sp.dmed)
        if sp.dran < dran_min:
            sp_min = sp
            dran_min = sp.dran
            # print("Minimum:",dran_min)
    sp_min.save_to(get_data(f"sampler_presets/sample_sphere_N_{N}.pkl"))

print("Generating the circle presets...")
for N in tqdm(SAMPLER_CIRCLE_PRESETS):
    sp = Sampler(N=N)
    sp.gen_circle()
    sp.save_to(get_data(f"sampler_presets/sample_circle_N_{N}.pkl"))
