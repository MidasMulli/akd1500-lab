const {contextBridge, ipcRenderer} = require('electron');
const actions=new Set(['state','pin','click-through','show','console','close']);
contextBridge.exposeInMainWorld('akdDesktop', Object.freeze({
  command(action) {
    if(!actions.has(action))return Promise.reject(new Error('Unknown desktop action'));
    return ipcRenderer.invoke('akd-desktop-action',action);
  },
  onState(callback) {
    if(typeof callback!=='function')return;
    ipcRenderer.on('akd-desktop-state',(_event,state)=>callback(state));
  }
}));
