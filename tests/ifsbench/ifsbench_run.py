#! @IFSBENCH_PYTHON@

# (C) Copyright 2024- ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from contextlib import nullcontext
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from typing import List, Dict, Union

import click
from pandas import DataFrame
from pydantic import field_serializer, field_validator, ConfigDict
import yaml

from ifsbench import (
    cli, DefaultApplication,
    DefaultArch, CpuConfiguration,
    Benchmark, ScienceSetup, TechSetup,
    Job, Launcher,
    PydanticConfigMixin, EnvHandler, DataFileStats, PydanticDataFrame)

from ifsbench.data import FetchHandler, NamelistHandler, NamelistOverride, RenameHandler, RenameMode
from ifsbench.validation import FrameCloseValidation

class EclandResult(PydanticConfigMixin):
    """
    Ecland result class that can be serialised using the ConfigMixin approach.
    """

    #: Numerical results of the run, stored as DataFrames (with corresponding
    #: property name).
    frames: Dict[str, PydanticDataFrame]

    #: Standard out of the run.
    stdout: str = None

    #: Standard error of the run.
    stderr: str = None

    #: Walltime of the run in seconds.
    walltime: float = None

    @classmethod
    def from_rundir(cls, run_dir: Path, **kwargs):
        """
        Build a result object from data in a run directory.

        Params
        ------
        run_dir: pathlib.Path
            The run directory.
        kwargs:
            If values are given for any of the EclandResult attributes, these
            values are used instead of the extracted data from the run directory.
        """

        # Copy the kwargs dict so we can modify it.
        kwargs = dict(kwargs)

        if 'stderr' in kwargs:
            # Extract the walltime from the stderr, if necessary.
            match = re.search(r'wall-time\s*:\s*([0-9\.]+)s', kwargs['stderr'])
            if match and 'walltime' not in kwargs:
                kwargs['walltime'] = float(match.group(1))

        # Gather the numerical data by parsing the netCDF files.
        frames = {}

        # ecLand writes output netCDF files of the form o_<some_name>.nc. Glob
        # all of them and parse them.
        paths = sorted([path for path in run_dir.glob('o_*.nc')])

        for path in paths:
            stats = DataFileStats(
                input_path=path,
                stat_dims=['lat', 'lon'],
                stat_names=['min', 'max', 'mean']
            )

            results = stats.get_stats()
            
            i = 1
            for result in results:
                frames[f"{path.name}_{i}"] = result

                i += 1

        if 'frames' not in kwargs:
            kwargs['frames'] = frames

        return cls(**kwargs)

class EclandScience(PydanticConfigMixin):
    """
    Science setup of the ecland benchmark.
    """

    #: URL to the forcing file.
    forcing_url: str

    #: URL to the soil file.
    soil_url: str

    #: URL to the surfclim file.
    surfclim_url: str 

    #: Path to the default namelist.
    namelist_url: str

    #: Path to the ecland build directory.
    build_dir: Path = None

    #: List of namelist overrides.
    namelists: List[NamelistOverride] = None

    #: List of custom environment overrides.
    env: List[EnvHandler] = None

    #: The default job setup.
    job: Job = None

class EclandTech(PydanticConfigMixin):
    """
    Tech setup of the ecland benchmark.
    """

    #: List of namelist overrides.
    namelists: List[NamelistOverride] = None

    #: List of custom environment overrides.
    env: List[EnvHandler] = None

    #: The number of GPUs that are used per task.
    gpus_per_task: int = None


class EclandConfig(PydanticConfigMixin):
    """
    This object describes the format of the YAML/JSON file from which the
    data is read.
    """
    science: Dict[str, EclandScience]
    tech: Dict[str, EclandTech]
    arch: Dict[str, DefaultArch]

def build_ecland_benchmark(science: EclandScience, tech: EclandTech) -> Benchmark:
    """
    Build an ifsbench Benchmark object from the ecland science and tech
    objects.
    """
    # Initial step is to
    #  * download the inidata tarball.
    #  * extract the inidata tarball.
    #  * Fetch the default namelist.
    data_handlers_init = [
        FetchHandler(source_url=science.forcing_url, target_path='forcing'),
        FetchHandler(source_url=science.soil_url, target_path='soilinit'),
        FetchHandler(source_url=science.surfclim_url, target_path='surfclim'),
        FetchHandler(source_url=science.namelist_url, target_path='namelist_template')
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
    
    return Benchmark(science = science_setup, tech=tech_setup)


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
@click.option('--launcher-flags', default=[], multiple=True, type=str,
              help='Additional flags that are passed to the launcher')
@click.option('--validate', type=click.Path(exists=True),
              help='Validate results against given result file.')
def from_yaml(yaml_path, science, tech, build_dir, run_dir, tasks, threads, 
    arch, launcher_flags, validate):
    """
    Run ecland benchmark from a file.
    """
    yaml_path = Path(yaml_path).resolve()

    with yaml_path.open('r') as f:
        yaml_data = yaml.safe_load(f)

    ecland_config = EclandConfig.from_config(yaml_data)

    science_input = ecland_config.science[science]
    tech_input = ecland_config.tech[tech]

    if build_dir:
        science_input.build_dir = Path(build_dir).resolve()

    benchmark = build_ecland_benchmark(science=science_input, tech=tech_input)
    
    job = science_input.job
    if tasks:
        job.tasks = tasks
    if threads:
        job.cpus_per_task = threads

    arch = ecland_config.arch.get(arch, ecland_config.arch['default'])

    if run_dir:
        run_dir = Path(run_dir).resolve()
        context = nullcontext(run_dir)
    else:
        context = TemporaryDirectory(dir=Path.cwd())

    with context as run_dir:
        run_dir = Path(run_dir)
        benchmark.setup_rundir(run_dir)

        bench_result = benchmark.run(
            run_dir=run_dir, 
            job=job,
            arch=arch,
            launcher_flags=launcher_flags
        )

        result = EclandResult.from_rundir(
            run_dir=run_dir,
            stdout=bench_result.stdout,
            stderr=bench_result.stderr,
        )

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
