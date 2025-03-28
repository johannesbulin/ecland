#! @IFSBENCH_PYTHON@

# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Dict, Union

import click
from pandas import DataFrame
import yaml

from ifsbench import (cli, DefaultApplication, Benchmark, ScienceSetup, TechSetup,
                      DefaultArch, Job, CpuConfiguration, MpirunLauncher, SrunLauncher,
                      PydanticConfigMixin, EnvHandler, ConfigMixin)
from ifsbench.data import DataHandler, ExtractHandler, RenameHandler, RenameMode, NamelistHandler, NamelistOverride
from ifsbench.validation import FrameCloseValidation

# We define default arches here, as arch serialisation isn't yet supported in
# ifsbench.
arches = {
    'default': DefaultArch(
        launcher=MpirunLauncher(),
        cpu_config=CpuConfiguration()
    ),
    'atos':  DefaultArch(
        launcher=SrunLauncher(),
        cpu_config=CpuConfiguration(
            sockets_per_node=2,
            cores_per_socket=64,
            threads_per_core=2,
            gpus_per_node=0
        )
    ),
    'lumi-c':  DefaultArch(
        launcher=SrunLauncher(),
        cpu_config=CpuConfiguration(
            sockets_per_node=2,
            cores_per_socket=64,
            threads_per_core=2,
            gpus_per_node=0
        ),
        account='project_465000527',
        partition='small'
    )
}

def parse_netcdf(path):
    """
    TODO: Move parts of netcdf parsing into ifsbench itself.

    Parse an ecland netcdf4 file and convert it into a variable_name/frame
    dictionary.
    Each frame holds the min/max/mean values, calculated for each level
    over the latitudes/longitudes.
    """
    import netCDF4
    import numpy

    result = {}
    rootgrp = netCDF4.Dataset(path, 'r')

    for var_name, value in rootgrp.variables.items():
        n_value = numpy.array(value[:])

        # TODO: What kind of output do we expect from ecland? For demo purposes
        # only (nlev, nlat, nlon) datasets are used.
        if n_value.ndim != 3:
           continue

        mean = n_value.mean(axis=(1,2))
        min = n_value.min(axis=(1,2))
        max = n_value.max(axis=(1,2))

        frame = DataFrame(
            numpy.array([mean, min, max]).T,
            index = [f'Level {l}' for l in range(n_value.shape[0])],
            columns = ['mean', 'min', 'max']
        )

        result[var_name] = frame

    return result

@dataclass
class EclandResult(ConfigMixin):
    """
    Ecland result class that can be serialised using the ConfigMixin approach.
    """

    # Numerical results of the run, stored as DataFrames (with corresponding
    # property name).
    frames: Dict[str, DataFrame]

    # Log of the run.
    log: str = None

    # Walltime of the run in some yet-to-be-specified unit.
    walltime: float = None

    @classmethod
    def from_rundir(cls, run_dir):
        frames = {}

        # Just open the o_fix.nc and o_gg.nc result files and get all the data
        # out of them.
        # TODO: Do we need more/other results?
        paths = [run_dir/'o_fix.nc', run_dir/'o_gg.nc']

        for path in paths:
            result = parse_netcdf(path)
            frames = {**frames, **result}

        # TODO: No logs or walltimes are added yet.

        return cls(frames=frames)

    def dump_config(
        self, with_class: bool = False
    ) -> Dict[str, Union[str, float, int, bool, List]]:

        # Serialise the result. We must use `to_dict(orient='split')` to keep
        # the column order of the frames!
        config = {
            'log': self.log,
            'walltime': self.walltime,
            'frames': {x: y.to_dict(orient='split') for x,y in self.frames.items()}
        }
        return config

    @classmethod
    def from_config(
        cls, config: Dict[str, Union[str, float, int, bool, List, None]]
    ) -> 'PydanticConfigMixin':
        config = dict(config)
        config['frames'] = {x: DataFrame(**y) for x,y in config['frames'].items()}

        return cls(**config)

class EclandScience(PydanticConfigMixin):
    """
    Science setup of the ecland benchmark.
    """

    # Path to the input tarball.
    input_archive: Path

    # Path to the ecland build directory.
    build_dir: Path = None

    # List of namelist overrides.
    namelists: List[NamelistOverride] = None

    # List of custom environment overrides.
    env: List[EnvHandler] = None

    # The default job setup.
    job: Job = None

class EclandTech(PydanticConfigMixin):
    """
    Tech setup of the ecland benchmark.
    """

    # List of namelist overrides.
    namelists: List[NamelistOverride] = None

    # List of custom environment overrides.
    env: List[EnvHandler] = None

    # The number of GPUs that are used per task.
    gpus_per_task: int = None


class EclandBenchmark(Benchmark):
    def __init__(self, science, tech):

        # Initial step is to extract the data tarball. Then rename `input` (the used
        # Fortran namelist) to `namelist_template` as this one will be modified later.
        data_handlers_init = [
            ExtractHandler.from_config(config={'archive_path': str(science.input_archive)}),
            RenameHandler(pattern='input$', repl='namelist_template', mode=RenameMode.MOVE)
        ]

        # At runtime, copy the original namelist back to `input`.
        data_handlers_runtime = [
            RenameHandler(pattern='namelist_template', repl='input', mode=RenameMode.COPY)
        ]

        # If namelist overrides are specified, also run them at runtime.
        if science.namelists:
            data_handlers_runtime.append(NamelistHandler('namelist_template', 'input', science.namelists))


        env_handlers = []

        if science.env:
            env_handlers += science.env

        application = DefaultApplication(
            command = [str(science.build_dir/'bin/ecland-master')],
        )

        science_setup = ScienceSetup(
            data_handlers_init = data_handlers_init,
            data_handlers_runtime = data_handlers_runtime,
            env_handlers = env_handlers,
            application = application
        )


        data_handlers_runtime = []

        if tech.namelists:
            data_handlers_runtime.append(NamelistHandler(
                input_path='namelist_template',
                output_path='input',
                overrides=tech.namelists
            ))

        env_handlers = []

        if tech.env:
            env_handlers += science.env

        tech_setup = TechSetup(
            data_handlers_runtime = data_handlers_runtime,
            env_handlers = env_handlers
        )

        super().__init__(science = science_setup, tech=tech_setup)

# class EclandEnsembleBenchmark(EclandBenchmark):
#     def __init__(self, science, tech, ensemble_size):
#         super().__init__(science, tech)

#         self.ensemble_size = ensemble_size

#     def setup_rundir(self,
#         run_dir: Path,
#         force: bool = False
#     ):

#         for i in range(self.ensemble_size):
#             super().setup_rundir(run_dir/f'run_{i}', force)

#             # Perturb ini


#     def run(self,
#         run_dir: Path,
#         job: Job,
#         arch: Optional[Arch] = None,
#         launcher: Optional[Launcher] = None,
#         launcher_flags: Optional[List[str]] = None
#     ):
#         results = []

#         for i in range(self.ensemble_size):
#             result = self.run(
#                 run_dir/f'run_{i}',
#                 job,
#                 arch,
#                 launcher,
#                 launcher_flags
#             )

#             result.append(result)

class EclandConfig(PydanticConfigMixin):
    science: Dict[str, EclandScience]
    tech: Dict[str, EclandTech]
#    arch: List[DefaultArch] = None


@cli.command('from_yaml', context_settings={"auto_envvar_prefix": "IFSBENCH"})
@click.argument('yaml-path', type=click.Path(exists=True))
@click.argument('science', type=str)
@click.option('--build-dir', type=click.Path(exists=True))
@click.option('--tech', type=str, default='default',)
@click.option('--run-dir', type=click.Path(), default=None,
              help='Run directory for the tests (temporary directory by default)')
@click.option('--tasks', type=int, default=None,
              help='Number of tasks to run')
@click.option('--threads', type=int, default=None,
              help='Number of threads to use')
@click.option('--arch', default=None, type=str,
              help='The architecture to use.')
@click.option('--validate', type=click.Path(exists=True),
              help='Validate results against given result file.')
def from_yaml(yaml_path, science, tech, build_dir, run_dir, tasks, threads, arch, validate):
    """
    Run ecland benchmark from a file.
    """
    yaml_path = Path(yaml_path).resolve()

    if build_dir:
        build_dir = Path(build_dir).resolve()

    with yaml_path.open('r') as f:
        yaml_data = yaml.safe_load(f)

    ecland_config = EclandConfig.from_config(yaml_data)

    science_input = ecland_config.science[science]
    tech_input = ecland_config.tech[tech]

    benchmark = EclandBenchmark(science = science_input, tech = tech_input)

    job = science_input.job
    if tasks:
        job.tasks = tasks
    if threads:
        job.cpus_per_task = threads

    arch = arches.get(arch, arches['default'])

    if run_dir:
        run_dir = Path(run_dir).resolve()
        context = nullcontext(run_dir)
    else:
        context = TemporaryDirectory(dir=Path.cwd())

    with context as run_dir:
        run_dir = Path(run_dir)
        benchmark.setup_rundir(run_dir)
        bench_result = benchmark.run(run_dir, job, arch)

        result = EclandResult.from_rundir(run_dir)

        with (run_dir/'result.yaml').open('w') as f:
            yaml.dump(result.dump_config(), f)

        if validate:
            validator = FrameCloseValidation(atol=0, rtol=0)
            with Path(validate).open('r') as f:
                reference = EclandResult.from_config(yaml.safe_load(f))

            if set(result.frames.keys()) != set(reference.frames.keys()):
                raise RuntimeError("Results do not hold the same frames!")

            for key in result.frames.keys():
                frame = result.frames[key]
                frame_ref = reference.frames[key]

                equal, mismatch = validator.compare(frame, frame_ref)

                if not equal:
                    raise RuntimeError("Results not equal!")

@cli.command('validate', context_settings={"auto_envvar_prefix": "IFSBENCH"})
@click.argument('result', type=click.Path(exists=True))
@click.argument('reference', type=click.Path(exists=True))
def validate(result, reference):
    """
    Compare two ecland result files and check for bit-identicality.
    """
    validator = FrameCloseValidation(atol=0, rtol=0)

    with Path(result).open('r') as f:
        result = EclandResult.from_config(yaml.safe_load(f))

    with Path(reference).open('r') as f:
        reference = EclandResult.from_config(yaml.safe_load(f))

    if set(result.frames.keys()) != set(reference.frames.keys()):
        raise RuntimeError("Results do not hold the same frames!")

    for key in result.frames.keys():
        frame = result.frames[key]
        frame_ref = reference.frames[key]

        equal, mismatch = validator.compare(frame, frame_ref)

        if not equal:
            raise RuntimeError("Results not equal!")

if __name__ == "__main__":
    cli(auto_envvar_prefix='IFSBENCH')