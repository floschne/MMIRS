import time

from loguru import logger
# TODO these util classes require knowledge how to use and do not cover or prevent error cases
from typing import Optional, Dict, List, Tuple


class TimingSession(object):
    def __init__(self):
        self.measurements: Dict[int, Dict[str, float]] = dict()
        self.measurements_stack: List[Tuple[str, float]] = list()

    def start_timing_measurement(self, name) -> None:
        if len(self.measurements_stack) != 0 and name == self.measurements_stack[-1][0]:
            logger.error(f"Timing with name {name} already started!")
            raise KeyError(f"Timing with name {name} already started!")

        # push name and start time
        if len(self.measurements_stack) > 0:
            caller = self.measurements_stack[-1][0]
            self.measurements_stack.append((f"{caller}>{name}", time.time()))
        else:
            self.measurements_stack.append((name, time.time()))

    def stop_timing_measurement(self) -> float:
        # pop name and start time
        current_level = len(self.measurements_stack)
        name, start = self.measurements_stack.pop()
        # stop time
        stop = time.time() - start
        # save in measurements dict
        if current_level not in self.measurements:
            self.measurements[current_level] = dict()
        self.measurements[current_level][name] = stop

        return stop

    def get_measurements(self) -> Dict[int, Dict[str, float]]:
        return {k: v for k, v in sorted(self.measurements.items(), key=lambda i: i[0], reverse=False)}


class MMIRSTimer(object):
    __singleton = None

    def __new__(cls, *args, **kwargs):
        if cls.__singleton is None:
            logger.info("Instantiating MMIRSTimer ...")
            cls.__singleton = super(MMIRSTimer, cls).__new__(cls)

            cls.current_timing_session: Optional[TimingSession] = None

        return cls.__singleton

    def start_new_timing_session(self) -> TimingSession:
        self.current_timing_session = TimingSession()
        logger.debug(f"Starting new TimingSession!")
        return self.current_timing_session

    def start_measurement(self, name):
        if self.current_timing_session is None:
            self.start_new_timing_session()
        self.current_timing_session.start_timing_measurement(name)

    def stop_measurement(self):
        if self.current_timing_session is None:
            logger.error(f"TimingSession did not start yet!")
            raise KeyError(f"TimingSession did not start yet!")
        self.current_timing_session.stop_timing_measurement()

    def get_current_timing_session(self) -> TimingSession:
        return self.current_timing_session
