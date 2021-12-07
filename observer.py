from __future__ import annotations

import string
import threading
from abc import ABC, abstractmethod
from typing import List

import speech_generator


class Subject(ABC):

    @abstractmethod
    def attach(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        pass

    @abstractmethod
    def notify(self) -> None:
        pass


class Person(Subject):

    _name: string = "Unbekannt"
    _has_mask: bool = True

    _observers: List[Observer] = []

    def attach(self, observer: Observer) -> None:
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    def notify(self) -> None:

        for observer in self._observers:
            observer.update(self)

    def update_name(self, newname) -> None:

        if newname != self._name:
            self._name = newname
            self.notify()
    
    def update_mask(self, status) -> None:

        if status != self._has_mask:
            self._has_mask = status
            self.notify()


class Observer(ABC):

    @abstractmethod
    def update(self, subject: Subject) -> None:
        pass


class MaskObserver(Observer):
    def update(self, subject: Subject) -> None:
        if subject._has_mask is False:
            t = threading.Thread(target=speech_generator.generate_output_speech, args=[subject._name])
            t.start()