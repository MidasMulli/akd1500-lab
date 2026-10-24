"""Adversarial write ranges: touch, crossing, zero-size and overflow cases."""
import shutil,subprocess,tempfile,unittest
from pathlib import Path
class GuardRanges(unittest.TestCase):
    def test_address_boundaries_and_wrap_cannot_escape_record_guard(self):
        compiler=shutil.which('g++') or shutil.which('clang++');self.assertIsNotNone(compiler)
        source='''#include "coprocessor/record_range.hpp"
#include <cassert>
int main(){
 assert(!record_ranges_overlap(0,100,100,200));
 assert(!record_ranges_overlap(200,1,100,200));
 assert(!record_ranges_overlap(150,0,100,200));
 assert(record_ranges_overlap(99,2,100,200));
 assert(record_ranges_overlap(100,1,100,200));
 assert(record_ranges_overlap(199,2,100,200));
 assert(record_ranges_overlap(0,300,100,200));
 assert(record_ranges_overlap(0xffffffffu,0xffffffffffffffffULL,100,200));
 assert(record_ranges_overlap(0x2000013f,2,0x20000140,0x200dc140));
 assert(!record_ranges_overlap(0x200dc140,512,0x20000140,0x200dc140));
}'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'test.cpp').write_text(source)
            subprocess.run([compiler,'-std=c++17','-I',str(Path(__file__).resolve().parents[1]),str(p/'test.cpp'),'-o',str(p/'test')],check=True,capture_output=True)
            subprocess.run([str(p/'test')],check=True,capture_output=True)
if __name__=='__main__':unittest.main()
