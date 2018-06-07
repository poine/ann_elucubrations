#! /usr/bin/env python
# -*- coding: utf-8 -*-

##
## compare legacy and angular oboservation
##
## ./dql__gym_pendulum.py --env pendulum-legacy-v0    --render-env --train --summary-dir results_pendulum/run_legacy --max-episodes 200 --save --save_dir saves_pendulum/legacy
## ./dql__gym_pendulum.py --env pendulum-ang-obs-v0 --render-env --train --summary-dir results_pendulum/run_ang    --max-episodes 200 --save --save_dir saves_pendulum/ang
##
## ./dql__gym_pendulum.py --env pendulum-legacy-v0    --render-env --test --load --load_dir saves_pendulum/legacy

##
## test training restart
##
## ./dql__gym_pendulum.py --env pendulum-legacy-v0  --train --summary-dir results_pendulum/run_restart_training --max-episodes 1 --save --save_dir saves_pendulum/restart_training
## ./dql__gym_pendulum.py --env pendulum-legacy-v0  --test --summary-dir results_pendulum/run_restart_training --max-episodes 5 --load --load_dir saves_pendulum/restart_training
## ./dql__gym_pendulum.py --env pendulum-legacy-v0  --test --summary-dir results_pendulum/run_restart_training --max-episodes 5 --load --load_dir saves_pendulum/restart_training

if __name__ == '__main__':

    pass
