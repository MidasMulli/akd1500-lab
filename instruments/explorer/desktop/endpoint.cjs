function makeEndpoint(value='http://127.0.0.1:8892/'){
  const base=new URL(value);
  if(base.protocol!=='http:'||!['localhost','127.0.0.1'].includes(base.hostname)||base.port!=='8892'||base.pathname!=='/'||base.username||base.password)
    throw new Error('Only the known private explorer endpoint on port 8892 is allowed');
  const target=new URL('/',base.origin);
  // Presentation only. Do not carry arbitrary startup/query commands into the host.
  target.search=new URLSearchParams({view:'nodes',background:'0',connections:'1',desktop:'1'}).toString();
  return {origin:base.origin,target};
}
module.exports={makeEndpoint};
