// Reset only this study's failed, now-unowned replay installation. No inference.
#define main inherited_readout_entry
#include "readout_bench.cpp"
#undef main
int main(int argc,char**argv){try{need(argc==2,"cleanup output");LabDriver d;auto dev=akida::HardwareDevice::create(&d);std::vector<uint32_t>before,after;for(auto b:{0x20000,0x24000,0x28000})before.push_back(d.read32(LabDriver::REG+b));dev->unprogram();for(auto b:{0x20000,0x24000,0x28000})after.push_back(d.read32(LabDriver::REG+b));dump(std::string(argv[1])+".before",before);dump(std::string(argv[1])+".after",after);for(auto v:after)need((v&1)==0,"DMA run bit clear after owned reset");std::cout<<"{\"status\":\"OWNED_REPLAY_RESET\",\"inferences\":0,\"dma_run_bits_clear\":true}\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
