// Adapted from the existing tools/subconscious-viz/electron transparent shell.
// This host has no SDK, model-writing capability or automatic experiment start.
const {app,BrowserWindow,Menu,globalShortcut,screen,ipcMain,shell}=require('electron');
const path=require('node:path');
const {makeEndpoint}=require('./endpoint.cjs');
const {origin,target}=makeEndpoint(process.env.AKD_EXPLORER_URL);
const restoreKey='CommandOrControl+Alt+Shift+N';
let win,pass=false,pinned=true,restoreRegistered=false;
const state=()=>({clickThrough:pass,alwaysOnTop:pinned,restoreRegistered,restoreShortcut:process.platform==='darwin'?'⌘⌥⇧N':'Ctrl+Alt+Shift+N'});
function publish(){
  if(win&&!win.isDestroyed())win.webContents.send('akd-desktop-state',state());
  console.log(JSON.stringify({kind:'OVERLAY_STATE',...state()}));
}
function clickThrough(value){
  if(value&&!restoreRegistered)throw new Error('Restore shortcut unavailable; click-through stays disabled');
  pass=Boolean(value);win.setIgnoreMouseEvents(pass,{forward:true});publish();rebuildMenu();
}
function show(){
  clickThrough(false);win.show();win.focus();
}
function sizeWindow(width,height){
  const area=screen.getDisplayMatching(win.getBounds()).workArea;
  win.setBounds({x:area.x+Math.round((area.width-Math.min(width,area.width))/2),y:area.y+Math.round((area.height-Math.min(height,area.height))/2),width:Math.min(width,area.width),height:Math.min(height,area.height)});
}
function rebuildMenu(){
  const controls=[
    {label:'Show network / restore interaction',click:show},
    {label:'Always on top',type:'checkbox',checked:pinned,click:()=>{pinned=!pinned;win.setAlwaysOnTop(pinned,'floating');publish();rebuildMenu();}},
    {label:'Click through to desktop',type:'checkbox',checked:pass,enabled:restoreRegistered,click:()=>clickThrough(!pass)},
    {type:'separator'},
    {label:'Compact · 840 × 560',click:()=>sizeWindow(840,560)},
    {label:'Wide · 1180 × 740',click:()=>sizeWindow(1180,740)},
    {label:'Large · 1480 × 900',click:()=>sizeWindow(1480,900)},
    {label:'Center window',click:()=>win.center()},
    {type:'separator'},
    {label:'Open full console in browser',click:()=>shell.openExternal(origin+'/')},
    {label:'Reload viewer',accelerator:'CommandOrControl+R',click:()=>{clickThrough(false);win.reload();}},
    {role:'quit'}
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {label:'AKD Network',submenu:controls},
    {label:'Edit',submenu:[{role:'copy'},{role:'paste'},{role:'selectAll'}]},
    {label:'Window',submenu:[{role:'minimize'},{role:'close'}]}
  ]));
}
async function createWindow(){
  const area=screen.getPrimaryDisplay().workArea;
  win=new BrowserWindow({
    title:'AKD Network Explorer — Desktop',width:Math.min(1180,area.width),height:Math.min(740,area.height),
    x:area.x+Math.round((area.width-Math.min(1180,area.width))/2),y:area.y+Math.round((area.height-Math.min(740,area.height))/2),
    frame:false,transparent:true,backgroundColor:'#00000000',hasShadow:false,
    alwaysOnTop:true,resizable:false,movable:true,minimizable:true,maximizable:false,fullscreenable:false,show:false,
    webPreferences:{preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true,partition:'akd-network-overlay'}
  });
  win.setAlwaysOnTop(true,'floating');
  if(process.platform==='darwin')win.setVisibleOnAllWorkspaces(true,{visibleOnFullScreen:true});
  const session=win.webContents.session;
  session.setPermissionRequestHandler((_contents,_permission,callback)=>callback(false));
  session.setPermissionCheckHandler(()=>false);
  win.webContents.setWindowOpenHandler(()=>({action:'deny'}));
  win.webContents.on('will-navigate',(event,url)=>{if(new URL(url).origin!==origin)event.preventDefault();});
  win.webContents.on('will-attach-webview',event=>event.preventDefault());
  win.webContents.on('page-title-updated',event=>event.preventDefault());
  win.webContents.on('did-finish-load',()=>publish());
  win.webContents.on('did-fail-load',(_event,code,description)=>console.error(JSON.stringify({kind:'LOAD_FAILED',code,description,origin})));
  win.on('closed',()=>{win=null;app.quit();});
  win.once('ready-to-show',()=>win.show());
  restoreRegistered=globalShortcut.register(restoreKey,()=>pass?show():clickThrough(true));
  rebuildMenu();
  await win.loadURL(target.href);
  console.log(JSON.stringify({kind:'DESKTOP_OVERLAY_READY',pid:process.pid,origin,electron:process.versions.electron,backgroundColor:win.getBackgroundColor(),bounds:win.getBounds(),...state()}));
}
ipcMain.handle('akd-desktop-action',(event,action)=>{
  if(!win||event.sender!==win.webContents||new URL(event.senderFrame.url).origin!==origin)throw new Error('Untrusted desktop command');
  switch(action){
    case 'state':break;
    case 'pin':pinned=!pinned;win.setAlwaysOnTop(pinned,'floating');publish();rebuildMenu();break;
    case 'click-through':clickThrough(!pass);break;
    case 'show':show();break;
    case 'console':shell.openExternal(origin+'/');break;
    case 'close':app.quit();break;
    default:throw new Error('Unknown desktop command');
  }
  return state();
});
app.setName('AKD Network Explorer');
app.setPath('userData',path.join(app.getPath('appData'),'AKD Network Explorer'));
if(!app.requestSingleInstanceLock()){app.quit();}
else{
  app.on('second-instance',()=>{if(win)show();});
  app.whenReady().then(createWindow).catch(error=>{console.error(error);app.quit();});
  app.on('window-all-closed',()=>app.quit());
  app.on('will-quit',()=>globalShortcut.unregisterAll());
}
