#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Game Time Module

"Game Time Module"

__title__ = 'gametime'

import time as _time

class GameClock:
    "Manages time in a game."
    __slots__ = ('game_ticks_per_second', 'game_tick', 'speed',
                 'clock_time', 'virtual_time', 'game_time',
                 'game_frame_count', 'real_time_passed',
                 'real_time', 'started', 'paused', 'between_frame',
                 'fps', 'fps_sample_start_time', 'fps_sample_count',
                 'average_fps')
    def __init__(self, game_ticks_per_second: int|float=20):
        """Create a Game Clock object.
        
        game_ticks_per_second -- The number of logic frames a second.
        
        """
        
        self.game_ticks_per_second = game_ticks_per_second
        self.game_tick = 1 / self.game_ticks_per_second        
        self.speed = 1
        
        self.clock_time = 0
        self.virtual_time = 0
        self.game_time = 0
        self.game_frame_count = 0
        self.real_time_passed = 0
        
        self.real_time = self.get_real_time()
        self.started = False
        self.paused = False        
        self.between_frame = 0
        
        self.fps = 0
        self.fps_sample_start_time = 0
        self.fps_sample_count = 0
        self.average_fps = 0
    
    def start(self) -> None:
        "Starts the Game Clock. Must be called once."
        if self.started:
            return
        
        self.clock_time = 0
        self.virtual_time = 0
        self.game_time = 0
        self.game_frame_count = 0
        self.real_time_passed = 0
        
        self.real_time = self.get_real_time()
        self.started = True
        
        self.fps = 0
        self.fps_sample_start_time = self.real_time
        self.fps_sample_count = 0
        
    def set_speed(self, speed: int|float) -> None:
        """Sets the speed of the clock.
        
        speed -- A time factor (1 is normal speed, 2 is twice normal)
        
        """
        if speed < 0:
            raise ValueError('Negative speeds not supported')
        self.speed = speed
    
    def pause(self) -> None:
        "Pauses the Game Clock."
        self.paused = True
    
    def unpause(self) -> None:
        "Un-pauses the Game Clock."
        self.paused = False
    
    def get_real_time(self) -> float:# pylint: disable=no-self-use
        """Returns the real time, as reported by the system clock.
        This method may be overriden."""

        return _time.time()
    
    def get_fps(self) -> tuple[int, int|float]:
        """Retrieves the current frames per second as a tuple containing
        the fps and average fps over a second."""
        return self.fps, self.average_fps
    
    def get_between_frame(self) -> int|float:
        """Returns the interpolant between the previous game tick and the
        next game tick."""
        
        return self.between_frame
    
    def update(self, max_updates: int = 0) -> tuple[int, int|float]:
        """Advances time, must be called once per frame. Yields tuples of
        game frame count and game time.
        
        max_updates -- Maximum number of game time updates to issue.
        """
        
        assert self.started, "You must call 'start' before using a GameClock."
        
        real_time_now = self.get_real_time()
        
        self.real_time_passed = real_time_now - self.real_time
        self.real_time = real_time_now
        
        self.clock_time += self.real_time_passed
        
        if not self.paused:
            self.virtual_time += self.real_time_passed * self.speed
        
        update_count = 0
        while self.game_time + self.game_tick < self.virtual_time:       
            self.game_frame_count += 1
            self.game_time = self.game_frame_count * self.game_tick
            yield (self.game_frame_count, self.game_time)
            
            if max_updates and update_count == max_updates:
                break
        
        self.between_frame = ( self.virtual_time - self.game_time ) / self.game_tick
        
        if self.real_time_passed != 0:
            self.fps = 1 / self.real_time_passed
        else:
            self.fps = 0
        
        self.fps_sample_count += 1
        
        if self.real_time - self.fps_sample_start_time > 1:
            
            self.average_fps = self.fps_sample_count / (self.real_time - self.fps_sample_start_time)
            self.fps_sample_start_time = self.real_time
            self.fps_sample_count = 0

def test() -> None:
    "Test game clock"
    clock = GameClock(20) # AI is 20 frames per second
    clock.start()
    
    while clock.virtual_time < 2:
        for (frame_count, game_time) in clock.update():
            print(f'Game frame #{frame_count}, {game_time:2.4f}')
        
        virtual_time = clock.virtual_time
        print(f'\t{clock.between_frame*100:2.2f}% between game frame, time is {virtual_time:2.4f}')

        _time.sleep(0.2) # Simulate time to render frame

if __name__ == '__main__':    
    test()
