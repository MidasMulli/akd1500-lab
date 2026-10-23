"""Stateful integer/set reference, following tools.verify's independently checked rule."""
import math
import numpy as np


class Reference:
    def __init__(self, ram, n=1024):
        self.ram = list(map(int, ram)); self.n = n; self.u = 192
        self.stride = 4 + n//32; self.rot = self.cursor = 0
        self.weights = [sum(self.ram[2+j*self.stride+4+k] << (32*k) for k in range(n//32)) for j in range(192)]

    def forward(self, values, label):
        x = sum(1 << int(k) for k in np.flatnonzero(values))
        scores = [(w & x).bit_count() for w in self.weights]
        mix = lambda p: ((p >> 2) & 7) | ((p & 3) << 3)
        self.rot = (self.rot + sum(map(mix, scores))) & 31
        eligible = [j for j in range(label*64, (label+1)*64) if scores[j] > max(0, self.ram[2+j*self.stride+3] >> 16)]
        if eligible:
            win = max(eligible, key=lambda j: (scores[j], -j)); base = 2+win*self.stride; before = self.weights[win]
            remove = {k for k in range(self.n) if before >> k & 1 and not x >> k & 1}
            insert = {k for k in range(self.n) if x >> k & 1 and not before >> k & 1}
            count = min(self.ram[base+3] & 65535, len(remove), len(insert))
            chosen_r, chosen_i = [], []; word = self.cursor; burst = scanned = 0
            if count:
                while len(chosen_r) < count or len(chosen_i) < count:
                    for pool, chosen in [(remove, chosen_r), (insert, chosen_i)]:
                        possible = [z for z in pool if z//32 == word]
                        if len(chosen) < count and possible:
                            z = min(possible, key=lambda z: ((z % 32)-self.rot) % 32); chosen.append(z); pool.remove(z)
                    self.rot = (self.rot + mix(scores[-1])) & 31; scanned += 1
                    if len(chosen_r) == count and len(chosen_i) == count: break
                    word = (word+1) % (self.n//32); burst = 0 if word == 0 else (burst+1) % 4
                    assert scanned <= (self.n//32)*(count+1)
                self.cursor = (word+1) % (self.n//32)
                self.rot = (self.rot + min(3-burst, self.n//32-1-word)*mix(scores[-1])) & 31
                for z in chosen_r: self.weights[win] &= ~(1 << z)
                for z in chosen_i: self.weights[win] |= 1 << z
                swap = max(8, (self.ram[base+3] & 65535)-2)
                self.ram[base+3] = (min(scores[win], 24) << 16) | swap
                for z in range(self.n//32): self.ram[base+4+z] = (self.weights[win] >> (32*z)) & 0xffffffff
        return scores
