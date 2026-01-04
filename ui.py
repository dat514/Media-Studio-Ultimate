import base64

LOGO_SVG = """<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <circle cx="64" cy="64" r="60" fill="#ff0055" opacity="0.2"/>
  <circle cx="64" cy="64" r="45" fill="#ff0055" opacity="0.4"/>
  <path d="M64 20 L90 50 L64 80 L38 50 Z" fill="#ff0055"/>
  <circle cx="64" cy="64" r="20" fill="#0e1117"/>
  <path d="M58 58 L74 68 L58 78 Z" fill="#fff"/>
</svg>"""
LOGO_BASE64 = base64.b64encode(LOGO_SVG.encode()).decode()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Media Studio Ultimate 2.4</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
    :root { --primary:#ff0055; --bg:#0a0e17; --card:#151b2e; --text:#fff; --text-dim:#999; }
    * { box-sizing:border-box; margin:0; padding:0; user-select:none; }
    body { background:linear-gradient(135deg,#0a0e17,#1a0033); color:var(--text); font-family:'Segoe UI',sans-serif; min-height:100vh; padding:20px; overflow-x:hidden; }
    .container { max-width:1100px; margin:0 auto; padding-bottom:60px; }
    
    .header { text-align:center; margin-bottom:30px; }
    .logo-big { width:80px; filter:drop-shadow(0 0 20px var(--primary)); margin-bottom:10px; }
    h1 { margin:5px 0 20px; font-size:2.2em; background:linear-gradient(90deg,#ff0055,#ffaa00); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    
    .tabs { display:flex; justify-content:center; gap:10px; margin-bottom:20px; flex-wrap:wrap; }
    .tab { padding:10px 20px; border-radius:20px; background:rgba(255,255,255,0.05); cursor:pointer; font-weight:bold; transition:all 0.3s; border:1px solid transparent; }
    .tab.active { background:var(--primary); box-shadow:0 0 15px rgba(255,0,85,0.4); }
    .tab:hover:not(.active) { background:rgba(255,255,255,0.1); }

    .section { display:none; animation:fadeIn 0.5s; }
    .section.active { display:block; }
    @keyframes fadeIn { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }

    .input-box { width:100%; padding:18px; border-radius:14px; border:none; background:var(--card); color:var(--text); font-size:1.1em; box-shadow:0 6px 20px rgba(0,0,0,0.5); }
    .btn { background:var(--primary); padding:12px 24px; border:none; border-radius:10px; color:#fff; font-weight:bold; cursor:pointer; margin:5px; transition:all .3s; font-size:1em; display:inline-flex; align-items:center; gap:8px; justify-content:center; }
    .btn:hover { background:#cc0044; transform:translateY(-2px); box-shadow:0 10px 25px rgba(255,0,85,.6); }
    .btn-secondary { background:#333; }
    .btn-secondary:hover { background:#444; }
    .btn-sm { padding:6px 12px; font-size:0.85em; }
    
    .card { background:var(--card); padding:25px; border-radius:18px; margin:20px 0; box-shadow:0 12px 40px rgba(0,0,0,0.7); }
    
    .row { display:flex; gap:15px; flex-wrap:wrap; align-items:start; }
    .col { flex:1; min-width:250px; }
    .select-style, .input-style { padding:12px; border-radius:8px; background:#222; color:#fff; border:1px solid #444; font-size:1em; width:100%; margin-bottom:10px; }
    .label-title { display:block; margin-bottom:6px; font-weight:bold; color:var(--text-dim); font-size:0.9em; }

    .editor-container { position:relative; width:100%; height:450px; background:#000; border-radius:10px; overflow:hidden; display:flex; align-items:center; justify-content:center; border:2px solid #333; }
    .media-preview { max-width:100%; max-height:100%; display:block; pointer-events:none; }
    
    #crop-wrapper { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
    #crop-box { position:absolute; border:2px solid #ff0055; box-shadow:0 0 0 9999px rgba(0,0,0,0.7); cursor:move; min-width:20px; min-height:20px; pointer-events:auto; display:none; }
    
    .resize-handle { position:absolute; width:12px; height:12px; background:#fff; border:1px solid #ff0055; z-index:10; }
    .rh-nw { top:-6px; left:-6px; cursor:nw-resize; }
    .rh-ne { top:-6px; right:-6px; cursor:ne-resize; }
    .rh-sw { bottom:-6px; left:-6px; cursor:sw-resize; }
    .rh-se { bottom:-6px; right:-6px; cursor:se-resize; }
    
    .shape-circle, .shape-triangle { border-radius:50%; }
    
    .status-text { text-align:center; font-size:1.1em; margin-top:15px; color:var(--primary); min-height:25px; }

    ::-webkit-scrollbar { width:8px; }
    ::-webkit-scrollbar-track { background:rgba(0,0,0,0.2); }
    ::-webkit-scrollbar-thumb { background:var(--primary); border-radius:4px; }

    /* MODAL STYLES */
    .modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.8); z-index:1000; overflow-y:auto; backdrop-filter:blur(5px); }
    .modal-content { position:relative; background:var(--card); margin:5% auto; padding:25px; width:90%; max-width:1000px; border-radius:15px; box-shadow:0 0 50px rgba(0,0,0,0.5); border:1px solid #333; }
    .close-modal { position:absolute; top:15px; right:20px; font-size:28px; cursor:pointer; color:#888; transition:0.2s; }
    .close-modal:hover { color:#fff; }
    
    .fmt-search { width:100%; padding:15px; font-size:1.2em; border-radius:10px; border:1px solid #444; background:#0a0e17; color:#fff; margin-bottom:20px; }
    
    .fmt-category { margin-bottom:30px; }
    .fmt-cat-title { font-size:1.3em; margin-bottom:15px; color:var(--primary); border-bottom:1px solid #333; padding-bottom:5px; display:flex; align-items:center; gap:10px; }
    .fmt-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(100px, 1fr)); gap:10px; }
    .fmt-item { background:#222; padding:15px 5px; text-align:center; border-radius:8px; cursor:pointer; transition:all 0.2s; border:1px solid transparent; font-weight:bold; color:#ddd; }
    .fmt-item:hover { background:var(--primary); color:#fff; transform:translateY(-3px); box-shadow:0 5px 15px rgba(255,0,85,0.3); }

    .fmt-item span { display:block; font-size:0.8em; opacity:0.7; margin-top:3px; font-weight:normal; }

    /* SHORTENER STYLES */
    .url-group { display:flex; align-items:stretch; margin-bottom:5px; }
    .url-prefix { background:#333; color:#aaa; padding:18px; border-radius:14px 0 0 14px; border:1px solid #444; border-right:none; display:flex; align-items:center; font-size:1.1em; }
    .url-input { border-radius:0 14px 14px 0 !important; border-left:none !important; flex:1; }
    .helper-text { text-align:center; color:#888; font-size:0.85em; margin-bottom:20px; margin-top:5px;}
    
    .result-box { border:2px solid #ffaa00; background:rgba(255,170,0,0.05); padding:25px; border-radius:18px; margin-top:20px; text-align:center; display:none; animation:fadeIn 0.5s; }
    .result-link { font-size:1.8em; font-weight:bold; color:#00aaff; margin:15px 0; word-break:break-all; }
    .btn-copy { background:#ffaa00; color:#000; display:inline-flex; align-items:center; gap:8px; padding:12px 30px; border-radius:8px; border:none; font-weight:bold; cursor:pointer; transition:0.3s; font-size:1.1em; }
    .btn-copy:hover { background:#ffcc00; box-shadow:0 0 15px rgba(255,170,0,0.4); transform:translateY(-2px); }

    /* ADVANCED QR STYLES */
    .qr-container { display:flex; gap:20px; height:600px; }
    .qr-config { flex:1; background:var(--card); border-radius:15px; border:1px solid #333; overflow:hidden; display:flex; flex-direction:column; }
    .qr-preview { width:400px; background:var(--card); border-radius:15px; border:1px solid #333; padding:20px; display:flex; flex-direction:column; align-items:center; }
    
    .config-nav { display:flex; background:#222; border-bottom:1px solid #333; }
    .nav-item { flex:1; text-align:center; padding:15px 10px; cursor:pointer; color:#888; font-weight:bold; transition:0.2s; border-bottom:3px solid transparent; }
    .nav-item:hover { color:#fff; background:rgba(255,255,255,0.05); }
    .nav-item.active { color:var(--primary); border-bottom-color:var(--primary); background:rgba(255,0,85,0.05); }
    
    .config-body { flex:1; padding:20px; overflow-y:auto; }
    .config-tab { display:none; animation:fadeIn 0.3s; }
    .config-tab.active { display:block; }
    
    .shape-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:10px; }
    .shape-item { background:#1a1f2e; padding:15px; border-radius:10px; border:2px solid transparent; cursor:pointer; text-align:center; transition:0.2s; }
    .shape-item:hover { border-color:#555; transform:translateY(-2px); }
    .shape-item.selected { border-color:var(--primary); background:rgba(255,0,85,0.1); box-shadow:0 0 15px rgba(255,0,85,0.2); }
    .shape-icon { width:40px; height:40px; margin-bottom:5px; opacity:0.8; }
    
    .color-section { margin-bottom:20px; }
    .color-toggle { display:flex; background:#111; padding:4px; border-radius:8px; margin-bottom:15px; }
    .color-opt { flex:1; text-align:center; padding:8px; cursor:pointer; border-radius:6px; font-size:0.9em; }
    .color-opt.active { background:#333; color:#fff; font-weight:bold; }
    
    .preview-box { width:100%; aspect-ratio:1; background:#fff; border-radius:10px; display:flex; align-items:center; justify-content:center; margin-bottom:20px; position:relative; overflow:hidden; }
    .qr-img-display { max-width:90%; max-height:90%; }
    .qr-download-btn { width:100%; border:none; padding:15px; font-weight:bold; font-size:1.1em; background:var(--primary); color:#fff; border-radius:8px; cursor:pointer; transition:0.2s; }
    .qr-download-btn:hover { background:#e6004c; transform:translateY(-2px); box-shadow:0 5px 15px rgba(255,0,85,0.4); }
    .logo-upload-box { border:2px dashed #444; border-radius:10px; padding:30px; text-align:center; cursor:pointer; transition:0.2s; color:#888; }
    .logo-upload-box:hover { border-color:var(--primary); color:#fff; background:rgba(255,255,255,0.05); }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <img src="data:image/svg+xml;base64,""" + LOGO_BASE64 + """ " class="logo-big" alt="Logo">
        <div style="position:absolute; top:20px; right:20px;">
           <button class="btn btn-sm btn-secondary" onclick="updateLibs()">Update Core</button>
        </div>
        <h1>Media Studio Ultimate 2.4</h1>
        <div class="tabs">
            <div class="tab active" onclick="switchTab('downloader')" id="tab-btn-downloader">Downloader</div>
            <div class="tab" onclick="switchTab('converter')" id="tab-btn-converter">Converter</div>
            <div class="tab" onclick="switchTab('resizer')" id="tab-btn-resizer">Editor (Resize/Crop)</div>
            <div class="tab" onclick="switchTab('gifmaker')" id="tab-btn-gifmaker">Video to GIF</div>
            <div class="tab" onclick="switchTab('shortener')" id="tab-btn-shortener">Links & QR</div>
            <div class="tab" onclick="switchTab('bgremover')" id="tab-btn-bgremover">BG Remover(Beta)</div>
            <div class="tab" onclick="switchTab('waveauth')" id="tab-btn-waveauth">WaveAuth</div>
        </div>
    </div>

    <div id="downloader" class="section active">
        <input type="text" id="url" class="input-box" placeholder="Paste URL (YouTube, TikTok, Facebook)...">
        <div style="margin-top:15px"><button class="btn" onclick="dl_analyze()">Analyze</button></div>
        <div class="card" id="dl-card" style="display:none">
            <div class="row">
                <img id="dl-thumb" style="width:200px; border-radius:10px;">
                <div class="col">
                    <h3 id="dl-title">-</h3>
                    <p id="dl-author" style="color:#888">-</p>
                    <div style="margin:15px 0">
                        <label class="label-title">Download Type</label>
                        <select id="dl-type" class="select-style" onchange="updateDlOptions()">
                            <option value="video">Video (MP4/MKV)</option>
                            <option value="audio">Audio (MP3/WAV)</option>
                        </select>
                        

                        <div id="dl-opt-video" class="row" style="margin-top:10px;">
                            <div class="col">
                                <label class="label-title">Resolution</label>
                                <select id="dl-res" class="select-style">
                                    <option value="2160">4K (2160p)</option>
                                    <option value="1440">2K (1440p)</option>
                                    <option value="1080" selected>1080p</option>
                                    <option value="720">720p</option>
                                    <option value="480">480p</option>
                                    <option value="best">Best Available</option>
                                </select>
                            </div>
                            <div class="col">
                                <label class="label-title">FPS Limit</label>
                                <select id="dl-fps" class="select-style">
                                    <option value="60">60 FPS</option>
                                    <option value="30">30 FPS</option>
                                    <option value="best">Auto / Max</option>
                                </select>
                            </div>
                        </div>


                        <div id="dl-opt-audio" class="row" style="margin-top:10px; display:none;">
                            <div class="col">
                                <label class="label-title">Audio Format</label>
                                <select id="dl-audio-fmt" class="select-style" onchange="updateAudioFormat()">
                                    <option value="mp3">MP3</option>
                                    <option value="wav">WAV (Lossless)</option>
                                </select>
                            </div>
                            <div class="col" id="dl-bitrate-container">
                                <label class="label-title">Audio Quality</label>
                                <select id="dl-bitrate" class="select-style">
                                    <option value="320">High (320kbps)</option>
                                    <option value="192">Medium (192kbps)</option>
                                    <option value="128">Low (128kbps)</option>
                                </select>
                            </div>
                        </div>

                        <div id="dl-opt-cover" class="row" style="margin-top:15px; background:#1a1f2e; padding:15px; border-radius:10px; border:1px solid #333;">
                             <label class="label-title" style="width:100%">Cover Art (Optional)</label>
                             <div class="col" style="flex:0 0 auto;">
                                 <div id="cover-placeholder" style="width:100px; height:100px; border:2px dashed #444; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#666; font-size:2em; cursor:pointer; background:#222;" onclick="selectCoverArt()">+</div>
                                 <img id="cover-preview" style="width:100px; height:100px; object-fit:cover; border-radius:8px; display:none; border:1px solid #444;">
                             </div>
                             <div class="col">
                                 <div style="margin-bottom:5px; font-size:0.9em; color:#ddd;">Embed custom album art into your downloaded file.</div>
                                 <button class="btn btn-sm btn-secondary" onclick="selectCoverArt()">Select Image</button>
                                 <button class="btn btn-sm btn-secondary" onclick="clearCoverArt()" style="background:#442222; margin-left:5px;">Remove</button>
                                 <div style="font-size:0.8em; color:#888; margin-top:8px;">Supports JPG/PNG. Auto-resized to fit.</div>
                                 <input type="hidden" id="dl-cover-path">
                             </div>
                        </div>
                        

                        <div id="dl-playlist-info" style="display:none; margin-top:15px; padding:15px; background:#222; border-radius:10px; border-left:4px solid var(--primary);">
                            <div style="font-weight:bold; margin-bottom:5px; color:var(--primary);">📋 Playlist Detected</div>
                            <div id="pl-title" style="margin-bottom:3px;"></div>
                            <div id="pl-count" style="color:#888; font-size:0.9em;"></div>
                        </div>
                    </div>
                    
                    <button class="btn" onclick="dl_start()">Download Now</button>
                    <div id="dl-status" class="status-text"></div>
                    <div id="dl-track-progress" style="text-align:center; margin-top:10px; color:#888; display:none;"></div>
                </div>
            </div>
        </div>
    </div>

    <div id="converter" class="section">
        <div class="card">
             <div style="text-align:center; margin-bottom:20px;">
                <button class="btn" onclick="conv_add()">+ Add Files</button>
                <button class="btn btn-secondary" onclick="conv_clear()">Clear List</button>
             </div>
             
             <!-- Format Selection -->
             <div style="background:#222; padding:20px; border-radius:12px; margin-bottom:20px; text-align:center;">
                 <label class="label-title">Selected Format</label>
                 <div style="display:flex; justify-content:center; align-items:center; gap:15px;">
                     <div id="selected-fmt-display" style="font-size:2em; font-weight:bold; color:var(--primary)">MP4</div>
                     <button class="btn btn-sm" onclick="openFmtModal()">Change Format</button>
                 </div>
                 <input type="hidden" id="conv-fmt" value="mp4">
             </div>

             <div id="conv-list" style="max-height:250px; overflow-y:auto; margin-bottom:20px; background:#0a0e17; border-radius:10px; padding:10px;">
                <div style="text-align:center; color:#555; padding:20px;">No files added yet.</div>
             </div>
             
             <button class="btn" style="width:100%; font-size:1.2em; padding:15px;" onclick="conv_start()">Start Batch Conversion</button>
             <div id="conv-status" class="status-text"></div>
        </div>
    </div>


    <div id="fmt-modal" class="modal">
        <div class="modal-content">
            <span class="close-modal" onclick="closeFmtModal()">&times;</span>
            <h2 style="text-align:center; margin-bottom:20px;">Select Target Format</h2>
            <input type="text" id="fmt-search-input" class="fmt-search" placeholder="Search format (e.g. mp3, mov, image)..." onkeyup="filterFormats()">
            <div id="fmt-container">

            </div>
        </div>
    </div>


    <div id="resizer" class="section">
        <div class="card">
            <div class="row">
                <div class="col" style="flex:2">
                    <div class="editor-container" id="editor-view">
                        <span style="color:#555">Select media to edit</span>
                        <img id="edit-img" class="media-preview" style="display:none">
                        <video id="edit-vid" class="media-preview" style="display:none" controls></video>
                        
                        <div id="crop-wrapper">
                            <div id="crop-box" onmousedown="cropStartDrag(event)">
                                <div class="resize-handle rh-nw" onmousedown="cropStartResize(event, 'nw')"></div>
                                <div class="resize-handle rh-ne" onmousedown="cropStartResize(event, 'ne')"></div>
                                <div class="resize-handle rh-sw" onmousedown="cropStartResize(event, 'sw')"></div>
                                <div class="resize-handle rh-se" onmousedown="cropStartResize(event, 'se')"></div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col" style="flex:1">
                    <button class="btn btn-sm" style="width:100%; margin-bottom:10px" onclick="edit_file()">Open File</button>
                    
                    <div style="background:#222; padding:15px; border-radius:10px; margin-bottom:10px">
                        <label class="label-title">Resize Output (px)</label>
                        <div class="row">
                            <input type="number" id="rs-w" class="input-style" placeholder="W (auto)" style="width:48%">
                            <input type="number" id="rs-h" class="input-style" placeholder="H (auto)" style="width:48%">
                        </div>
                    </div>

                    <div style="background:#222; padding:15px; border-radius:10px; margin-bottom:10px">
                        <div class="row" style="justify-content:space-between">
                            <label class="label-title">Crop</label>
                            <button class="btn btn-sm btn-secondary" onclick="toggleCrop()">Toggle Visual</button>
                        </div>
                        <label class="label-title" style="margin-top:5px">Shape</label>
                        <select id="crop-shape" class="select-style" onchange="updateShape()">
                            <option value="rect">Rectangle</option>
                            <option value="circle">Circle / Oval</option>
                            <option value="triangle">Triangle (Beta)</option>
                        </select>
                        <div class="row" style="font-size:0.85em; display:none;">
                             <input id="cp-x" value="0"><input id="cp-y" value="0"><input id="cp-w" value="0"><input id="cp-h" value="0">
                        </div>
                    </div>

                    <button class="btn" style="width:100%" onclick="edit_process()">Save / Process</button>
                    <div id="edit-status" class="status-text"></div>
                </div>
            </div>
        </div>
    </div>

    <div id="gifmaker" class="section">
        <div class="card">
             <div class="row">
                <div class="col" style="flex:2">
                     <div class="editor-container" id="gif-view">
                        <span id="gif-ph" style="color:#555">Select video</span>
                        <video id="gif-vid" class="media-preview" style="display:none; pointer-events:auto;" controls></video>
                        <img id="gif-preview" class="media-preview" style="display:none">
                        
                         <div id="gif-crop-wrapper" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none;">
                            <div id="gif-crop-box" onmousedown="gifCropStartDrag(event)" style="position:absolute; border:2px solid #00ffaa; box-shadow:0 0 0 9999px rgba(0,0,0,0.7); cursor:move; min-width:20px; min-height:20px; pointer-events:auto; display:none;">
                                <div class="resize-handle rh-nw" onmousedown="gifCropStartResize(event, 'nw')"></div>
                                <div class="resize-handle rh-ne" onmousedown="gifCropStartResize(event, 'ne')"></div>
                                <div class="resize-handle rh-sw" onmousedown="gifCropStartResize(event, 'sw')"></div>
                                <div class="resize-handle rh-se" onmousedown="gifCropStartResize(event, 'se')"></div>
                            </div>
                        </div>
                     </div>
                </div>
                <div class="col">
                     <button class="btn btn-sm" style="width:100%" onclick="gif_file()">Open Video</button>
                     <div style="margin:15px 0">
                         <label class="label-title">Trim (Sec)</label>
                         <div class="row">
                             <input id="gif-start" class="input-style" placeholder="Start" value="0">
                             <input id="gif-end" class="input-style" placeholder="End">
                         </div>
                         <label class="label-title">Settings</label>
                         <div class="row">
                             <div class="col"><input id="gif-fps" class="input-style" value="15" title="FPS"></div>
                             <div class="col"><input id="gif-width" class="input-style" value="480" title="Width"></div>
                         </div>
                         <div style="background:#222; padding:10px; border-radius:8px; margin-top:10px">
                             <div class="row" style="justify-content:space-between">
                                 <label class="label-title">Crop</label>
                                 <button class="btn btn-sm btn-secondary" onclick="toggleGifCrop()">Toggle Visual</button>
                             </div>
                             <input id="gif-crop" class="input-style" placeholder="x:y:w:h">
                         </div>
                     </div>
                     <button class="btn" style="width:100%" onclick="gif_create()">Create GIF</button>
                     <div id="gif-status" class="status-text"></div>
                </div>
             </div>
        </div>
    </div>

    <div id="shortener" class="section">
        <div class="card">
            <h2 style="text-align:center; margin-bottom:30px; color:var(--primary);">URL Shortener</h2>
            
            <div style="max-width:700px; margin:0 auto;">
                <label class="label-title" style="text-align:center; font-size:1.1em; margin-bottom:10px;">Paste your long URL</label>
                <input type="text" id="short-url-input" class="input-box" placeholder="Paste a link (http:// or https://)..." style="margin-bottom:20px;">
                
                <label class="label-title" style="text-align:center; font-size:1.1em; margin-bottom:10px;">Custom short link (optional)</label>
                <div class="url-group">
                    <div class="url-prefix">is.gd/</div>
                    <input type="text" id="short-alias" class="input-box url-input" placeholder="your-custom-name">
                </div>
                <div class="helper-text">Only letters, numbers, - or _ (max 30 characters).</div>
                
                <button class="btn" onclick="shorten_url()" style="width:100%; font-size:1.2em; margin-top:10px; padding:15px;">Shorten URL</button>
                
                <div id="short-status" class="status-text"></div>

                <div id="short-result" class="result-box">
                    <div style="font-weight:bold; color:var(--text); margin-bottom:5px;">Your Shortened Link</div>
                    <div id="short-url-display" class="result-link"></div>
                    <button class="btn-copy" onclick="copyShortUrl()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2-2v1"></path></svg>
                        Copy Link
                    </button>
                </div>
            </div>
        </div>
        

        <div id="qr-section-inner" style="margin-top:40px; border-top:1px solid #333; padding-top:20px;">
             <h2 style="text-align:center; margin-bottom:20px; color:var(--primary);">Advanced QR Designer</h2>
             <div class="qr-container">
                 <!-- LEFT CONFIG -->
                <div class="qr-config">
                    <div class="config-nav">
                        <div class="nav-item active" onclick="switchQrTab('content')">Content</div>
                        <div class="nav-item" onclick="switchQrTab('shapes')">Shapes</div>
                        <div class="nav-item" onclick="switchQrTab('colors')">Colors</div>
                        <div class="nav-item" onclick="switchQrTab('logo')">Logo</div>
                    </div>
                    
                    <div class="config-body">

                        <div id="qt-content" class="config-tab active">
                            <label class="label-title">Website URL or Text</label>
                            <textarea id="avr-data" class="input-box" rows="5" placeholder="https://example.com" oninput="debounceAvr()"></textarea>
                        </div>
                        

                        <div id="qt-shapes" class="config-tab">
                            <label class="label-title">Data Modules</label>
                            <div class="shape-grid">
                                <div class="shape-item selected" onclick="selShape(this, 'square')">
                                    <div style="font-size:24px">⬛</div>
                                    <div>Square</div>
                                </div>
                                <div class="shape-item" onclick="selShape(this, 'circle')">
                                    <div style="font-size:24px">●</div>
                                    <div>Dot</div>
                                </div>
                                <div class="shape-item" onclick="selShape(this, 'rounded')">
                                    <div style="font-size:24px">▢</div>
                                    <div>Rounded</div>
                                </div>
                                <div class="shape-item" onclick="selShape(this, 'gapped')">
                                    <div style="font-size:24px">▪️</div>
                                    <div>Gapped</div>
                                </div>
                                 <div class="shape-item" onclick="selShape(this, 'vertical')">
                                    <div style="font-size:24px">|||</div>
                                    <div>Vertical</div>
                                </div>
                                 <div class="shape-item" onclick="selShape(this, 'horizontal')">
                                    <div style="font-size:24px">≡</div>
                                    <div>Horizontal</div>
                                </div>
                            </div>
                            <input type="hidden" id="avr-shape" value="square">
                        </div>
                        

                        <div id="qt-colors" class="config-tab">
                            <label class="label-title">Color Mode</label>
                            <div class="color-toggle">
                                <div class="color-opt active" onclick="switchColorMode('solid')" id="cm-btn-solid">Single Color</div>
                                <div class="color-opt" onclick="switchColorMode('gradient')" id="cm-btn-gradient">Gradient</div>
                            </div>
                            <input type="hidden" id="avr-color-mode" value="solid">
                            
                            <div id="cm-panel-solid">
                                 <label class="label-title">Foreground Color</label>
                                 <input type="color" id="avr-fill" value="#000000" class="input-style" style="height:50px; cursor:pointer;" onchange="debounceAvr()">
                            </div>
                            
                            <div id="cm-panel-gradient" style="display:none">
                                 <label class="label-title">Gradient Type</label>
                                 <select id="avr-grad-type" class="select-style" onchange="debounceAvr()">
                                     <option value="vertical">Linear Vertical (Top-Bottom)</option>
                                     <option value="horizontal">Linear Horizontal (Left-Right)</option>
                                     <option value="radial">Radial (Center-Edge)</option>
                                     <option value="square">Square (Center-Edge)</option>
                                 </select>
                                 
                                 <div class="row">
                                     <div class="col">
                                         <label class="label-title">Start / Center</label>
                                         <input type="color" id="avr-g-start" value="#ff0000" class="input-style" style="height:50px" onchange="debounceAvr()">
                                     </div>
                                     <div class="col">
                                         <label class="label-title">End / Edge</label>
                                         <input type="color" id="avr-g-end" value="#0000ff" class="input-style" style="height:50px" onchange="debounceAvr()">
                                     </div>
                                 </div>
                            </div>
                            
                            <div style="margin-top:20px; border-top:1px solid #333; padding-top:20px;">
                                 <label class="label-title">Background Color</label>
                                 <input type="color" id="avr-back" value="#ffffff" class="input-style" style="height:50px; cursor:pointer;" onchange="debounceAvr()">
                            </div>
                        </div>
                        

                        <div id="qt-logo" class="config-tab">
                             <label class="label-title">Upload Logo</label>
                             <div class="logo-upload-box" onclick="uploadAvrLogo()">
                                 <div style="font-size:3em">☁️</div>
                                 <div id="avr-logo-name">Click to Upload</div>
                             </div>
                             <button class="btn btn-sm btn-secondary" id="avr-rm-logo" onclick="rmAvrLogo(event)" style="margin-top:10px; display:none;">Remove Logo</button>
                        </div>
                    </div>
                </div>
                

                <div class="qr-preview">
                    <h3 style="margin-bottom:20px; color:#aaa;">Preview</h3>
                    <div class="preview-box">
                        <img id="avr-preview" class="qr-img-display" src="">
                        <div id="avr-loading" style="position:absolute; background:rgba(255,255,255,0.8); padding:10px 20px; border-radius:20px; color:#000; font-weight:bold; display:none;">Generating...</div>
                    </div>
                    <button class="qr-download-btn" onclick="downloadAvr()">
                         Download High Res
                    </button>
                    <div style="margin-top:10px; font-size:0.8em; color:#666;">
                        Generates transparent PNG with high quality vectors.
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div id="bgremover" class="section">
        <div class="card">
            <h2 style="text-align:center; margin-bottom:20px; color:var(--primary);">Background Remover (Beta)</h2>
            <div class="row">
                <div class="col" style="flex:2">
                    <div class="editor-container" style="background:#000 url('data:image/svg+xml;utf8,<svg width=\\'20\\' height=\\'20\\' xmlns=\\'http://www.w3.org/2000/svg\\'><rect x=\\'0\\' y=\\'0\\' width=\\'10\\' height=\\'10\\' fill=\\'%23333\\'/><rect x=\\'10\\' y=\\'10\\' width=\\'10\\' height=\\'10\\' fill=\\'%23333\\' /></svg>');">
                        <span id="bg-ph" style="color:#fff; background:rgba(0,0,0,0.5); padding:10px; border-radius:5px;">Select an image to remove background</span>
                        <img id="bg-preview" class="media-preview" style="display:none">
                        <img id="bg-result" class="media-preview" style="display:none; position:absolute; top:0; left:0; pointer-events:none;">
                         <canvas id="bg-canvas" width="800" height="600" style="position:absolute; top:0; left:0; width:100%; height:100%; display:none; cursor:crosshair; pointer-events:auto; touch-action:none; z-index:100;"></canvas>
                    </div>
                </div>
                <div class="col">
                     <button class="btn" style="width:100%" onclick="bg_open()">Open Image</button>
                     <div style="margin:20px 0; text-align:center;">
                         
                         <div style="margin-bottom:15px; text-align:left; background:#1a1f2e; padding:15px; border-radius:10px; border:1px solid #333;">
                             <label class="label-title">AI Model</label>
                             <select id="bg-model" class="select-style">
                                 <option value="isnet-general-use" selected>ISNet General (Best Balance)</option>
                                 <option value="u2net">U2Net (Standard)</option>
                                 <option value="u2net_human_seg">Human Segmentation</option>
                                 <option value="isnet-anime">Anime / Cartoon</option>
                             </select>
                             <div style="margin-top:12px;">
                                 <label style="display:flex; align-items:center; cursor:pointer;" title="Enable for better detailed edges like hair">
                                     <input type="checkbox" id="bg-alpha" style="width:18px; height:18px; margin-right:10px; accent-color:var(--primary);">
                                     <span style="font-size:1em; color:#ddd;">Enable Alpha Matting</span>
                                 </label>
                             </div>
                         </div>
                         
                         <!-- BRUSH TOOLS -->
                         <div id="bg-tools" style="display:none; margin-bottom:15px; text-align:left; background:#1a1f2e; padding:15px; border-radius:10px; border:1px solid #333;">
                             <label class="label-title">Manual Refine (Brush)</label>
                             
                             <div class="row" style="margin-bottom:10px; gap:5px;">
                                 <button class="btn btn-sm active" id="btn-tool-brush" onclick="setBgTool('restore')" style="flex:1; background:#444;">Restore</button>
                                 <button class="btn btn-sm" id="btn-tool-erase" onclick="setBgTool('erase')" style="flex:1; background:#222;">Erase</button>
                             </div>
                             
                             <label class="label-title" style="font-size:0.85em;">Brush Size</label>
                             <div class="row" style="align-items:center;">
                                 <input type="range" id="bg-brush-size" min="5" max="100" value="20" style="flex:1; margin-right:10px;">
                                 <b id="bg-brush-val" style="color:#ddd;">20</b>
                             </div>
                             
                             <div class="row" style="margin-top:10px; gap:5px;">
                                 <div class="col"><button class="btn btn-sm btn-secondary" onclick="bg_undo()" style="width:100%">Undo</button></div>
                                 <div class="col"><button class="btn btn-sm btn-secondary" onclick="bg_redo()" style="width:100%">Redo</button></div>
                             </div>
                         </div>
                         
                         <button class="btn" onclick="bg_process()" id="btn-bg-process" disabled>Remove Background</button>
                     </div>
                     <div id="bg-status" class="status-text"></div>
                     <button class="btn btn-secondary" onclick="bg_save()" id="btn-bg-save" style="width:100%; margin-top:20px;" disabled>Download Result</button>
                </div>
            </div>
        </div>
    </div>


    <div id="waveauth" class="section">
        <div class="card">
            <h2 style="text-align:center; margin-bottom:20px; color:var(--primary);">WaveAuth Audio Verifier v4.0</h2>
            <div class="row">

                <div class="col" style="flex:1; max-width:350px;">
                     <div style="background:#222; padding:15px; border-radius:12px; height:100%; max-height:600px; display:flex; flex-direction:column;">
                        <h4 style="margin-bottom:10px; color:#aaa;">Batch Queue</h4>
                        
                        <div style="display:flex; gap:5px; margin-bottom:10px;">
                            <button class="btn btn-sm" onclick="wa_addFiles()" style="flex:1;">+ Add Files</button>
                            <button class="btn btn-sm btn-secondary" onclick="wa_clearQ()">Clear</button>
                        </div>
                        
                        <div id="wa-queue" style="flex:1; overflow-y:auto; background:#151b2e; border-radius:8px; padding:10px; margin-bottom:15px;">
                            <div style="text-align:center; color:#555; font-size:0.9em;">Queue empty</div>
                        </div>
                        
                        <button class="btn" onclick="wa_processQ()" id="wa-btn-run" disabled>Start Batch Processing</button>
                        
                        <hr style="border-color:#333; margin:15px 0;">
                        
                        <h4 style="margin-bottom:10px; color:#aaa;">History Repository</h4>
                        

                        <div style="display:flex; justify-content:space-between; margin-bottom:10px; font-size:0.8em; color:#ddd; background:#111; padding:8px; border-radius:6px;">
                            <div style="text-align:center;">Files<br><b id="wa-dash-files">0</b></div>
                            <div style="text-align:center;">Avg Score<br><b id="wa-dash-score" style="color:var(--primary)">0.0</b></div>
                            <div style="text-align:center;">Avg Time<br><b id="wa-dash-time">0s</b></div>
                        </div>
                        
                        <div id="wa-history" style="flex:1; overflow-y:auto; background:#151b2e; border-radius:8px; padding:10px;">

                        </div>
                        <button class="btn btn-sm btn-secondary" onclick="wa_exportCSV()" style="margin-top:5px; width:100%;">Download Report (CSV)</button>
                        <button class="btn btn-sm btn-secondary" onclick="wa_clearHistory()" style="margin-top:5px; width:100%;">Clear History</button>
                        <button class="btn btn-sm btn-secondary" onclick="wa_clearTempImages()" style="margin-top:5px; width:100%; border:1px solid #444;">Clear Temp Images</button>
                     </div>
                </div>
                

                <div class="col" style="flex:2;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <h3 id="wa-filename" style="margin:0;">No file selected</h3>
                        <div>
                             <button class="btn btn-sm btn-secondary" onclick="wa_showRef()">Reference Examples</button>
                             <button class="btn btn-sm btn-secondary" onclick="wa_exportImg()" id="wa-btn-export" disabled>Save Spectrogram</button>
                        </div>
                    </div>
                    
                    <div class="editor-container" style="background:#000; height:300px; margin-bottom:20px; border:1px solid #333;">
                        <img id="wa-spectrogram" class="media-preview" style="display:none; width:100%; height:100%; object-fit:contain;">
                        <span id="wa-ph" style="color:#555;">Spectrogram will appear here</span>
                    </div>
                    
                    <div class="row">
                        <div class="col">
                            <div style="background:#222; padding:15px; border-radius:10px;">
                                <label class="label-title">Analysis Result (v4.0)</label>
                                <div id="wa-verdict" style="font-size:1.3em; font-weight:bold; color:var(--primary); margin-bottom:5px;">-</div>
                                <div id="wa-score" style="font-size:1.5em; color:#fff;">Score: -</div>
                                <div id="wa-reasons" style="margin-top:10px; font-size:0.9em; color:#ffaa00;">

                                </div>
                                <div style="margin-top:15px; font-size:0.8em; color:#888; border-top:1px solid #444; padding-top:10px; font-style:italic;">
                                    * Verdicts are reference only. Spectrogram validation is 100% accurate.
                                </div>
                            </div>
                        </div>
                        <div class="col">
                            <div style="background:#222; padding:15px; border-radius:10px;">
                                <label class="label-title">Technical Metrics</label>
                                <div style="font-size:0.95em; color:#ccc;">
                                    <div style="display:flex; justify-content:space-between;"><span>Bandwidth Util:</span> <span id="wa-mp3" style="color:#00aaff">-</span></div>
                                    <div style="display:flex; justify-content:space-between;"><span>Slope Metric:</span> <span id="wa-slope" style="color:#fff">-</span></div>
                                    <div style="display:flex; justify-content:space-between;"><span>Bandwidth:</span> <span id="wa-bw" style="color:#fff">-</span></div>
                                    <div style="display:flex; justify-content:space-between;"><span>Dyn Range:</span> <span id="wa-dr" style="color:#fff">-</span></div>
                                    <div style="display:flex; justify-content:space-between;"><span>Format:</span> <span id="wa-fmt" style="color:#fff">-</span></div>
                                    <div style="display:flex; justify-content:space-between;"><span>Duration:</span> <span id="wa-dur" style="color:#fff">-</span></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div id="wa-status" class="status-text" style="text-align:left; margin-left:5px;"></div>
                </div>
            </div>
        </div>
    </div>


    <div id="wa-ref-modal" class="modal">
        <div class="modal-content" style="max-width:1100px;">
            <span class="close-modal" onclick="document.getElementById('wa-ref-modal').style.display='none'">&times;</span>
            <h2 style="text-align:center; margin-bottom:20px;">WaveAuth Reference Examples</h2>
            <p style="text-align:center; color:#aaa; margin-bottom:20px;">Compare your result with these baselines.</p>
            <div class="row" style="text-align:center;">
                <div class="col">
                    <h3 style="color:#00ff88; margin-bottom:10px;">Example A</h3>
                    <img src="http://127.0.0.1:8000/stream?path=d:/Git%20Clone%20Folder/Media-Downloader/screenshots/E1.png" style="width:100%; border:2px solid #333; border-radius:8px;">
                     <p style="margin-top:5px; color:#888;">Example 1</p>
                </div>
                <div class="col">
                    <h3 style="color:#ff0055; margin-bottom:10px;">Example B</h3>
                     <img src="http://127.0.0.1:8000/stream?path=d:/Git%20Clone%20Folder/Media-Downloader/screenshots/E2.png" style="width:100%; border:2px solid #333; border-radius:8px;">
                     <p style="margin-top:5px; color:#888;">Example 2</p>
                </div>
            </div>
        </div>
    </div>
<script>
    function switchTab(id) {
        document.querySelectorAll('.section').forEach(e => e.classList.remove('active'));
        document.querySelectorAll('.tab').forEach(e => e.classList.remove('active'));
        document.getElementById(id).classList.add('active');
        document.getElementById('tab-btn-'+id).classList.add('active');
    }
    
    // --- FORMATS DATABASE ---
    const FORMATS_DB = {
        "Video": [
            {ext:"mp4", desc:"Universal Video"}, {ext:"mkv", desc:"Matroska"}, {ext:"avi", desc:"Standard"}, {ext:"mov", desc:"QuickTime"}, 
            {ext:"wmv", desc:"Windows Media"}, {ext:"flv", desc:"Flash"}, {ext:"webm", desc:"Web Video"}, {ext:"m4v", desc:"iTunes"}, 
            {ext:"mpg", desc:"MPEG-1/2"}, {ext:"mpeg", desc:"MPEG"}, {ext:"3gp", desc:"Mobile"}, {ext:"ts", desc:"Stream"}, 
            {ext:"vob", desc:"DVD"}, {ext:"ogv", desc:"Ogg"}, {ext:"m2ts", desc:"Blu-ray"}, {ext:"divx", desc:"DivX"}, 
            {ext:"asf", desc:"Advanced Stream"}, {ext:"f4v", desc:"Flash HD"}, {ext:"rm", desc:"RealMedia"}, {ext:"swf", desc:"Flash"}, 
            {ext:"mxf", desc:"Broadcast"}, {ext:"gif", desc:"Animated GIF (No Audio)"}
        ],
        "Audio": [
            {ext:"mp3", desc:"MPEG Audio"}, {ext:"wav", desc:"Waveform"}, {ext:"aac", desc:"Adv. Audio Coding"}, {ext:"flac", desc:"Lossless"}, 
            {ext:"ogg", desc:"Vorbis"}, {ext:"m4a", desc:"MPEG-4 Audio"}, {ext:"wma", desc:"Windows Audio"}, {ext:"alac", desc:"Apple Lossless"}, 
            {ext:"aiff", desc:"Mac Audio"}, {ext:"opus", desc:"High Quality"}, {ext:"mid", desc:"MIDI"}, {ext:"amr", desc:"Speech"}, 
            {ext:"ac3", desc:"Dolby Digital"}, {ext:"dts", desc:"Surround"}, {ext:"ape", desc:"Monkey Audio"}, {ext:"mka", desc:"Matroska Audio"}
        ],
        "Image": [
            {ext:"jpg", desc:"JPEG"}, {ext:"png", desc:"Portable Network"}, {ext:"gif", desc:"GIF Image"}, {ext:"webp", desc:"Web Picture"}, 
            {ext:"bmp", desc:"Bitmap"}, {ext:"tiff", desc:"TIFF"}, {ext:"ico", desc:"Icon"}, {ext:"svg", desc:"Vector"}, 
            {ext:"heic", desc:"High Eff."}, {ext:"tga", desc:"Targa"}, {ext:"dds", desc:"DirectDraw"}, {ext:"psd", desc:"Photoshop"}
        ]
    };
    
    // Init Visual Modal
    window.addEventListener('DOMContentLoaded', () => {
        renderFormatModal();
    });

    function renderFormatModal() {
        const c = document.getElementById('fmt-container');
        c.innerHTML = '';
        for(let [cat, items] of Object.entries(FORMATS_DB)) {
            const catDiv = document.createElement('div');
            catDiv.className = 'fmt-category';
            catDiv.innerHTML = `<div class="fmt-cat-title">${cat}</div>`;
            
            const grid = document.createElement('div');
            grid.className = 'fmt-grid';
            
            items.forEach(item => {
                const el = document.createElement('div');
                el.className = 'fmt-item';
                el.innerHTML = `${item.ext.toUpperCase()}<span>${item.desc}</span>`;
                el.onclick = () => selectFormat(item.ext);
                el.dataset.search = (item.ext + " " + item.desc + " " + cat).toLowerCase();
                grid.appendChild(el);
            });
            
            catDiv.appendChild(grid);
            c.appendChild(catDiv);
        }
    }
    
    function filterFormats() {
        const q = document.getElementById('fmt-search-input').value.toLowerCase();
        document.querySelectorAll('.fmt-item').forEach(el => {
            el.style.display = el.dataset.search.includes(q) ? 'block' : 'none';
        });
        // Hide empty categories
        document.querySelectorAll('.fmt-category').forEach(cat => {
            const children = Array.from(cat.querySelectorAll('.fmt-item'));
            const hasVisible = children.some(c => c.style.display !== 'none');
            cat.style.display = hasVisible ? 'block' : 'none';
        });
    }

    function openFmtModal() {
        document.getElementById('fmt-modal').style.display = 'block';
        document.getElementById('fmt-search-input').focus();
    }
    function closeFmtModal() {
        document.getElementById('fmt-modal').style.display = 'none';
    }
    function selectFormat(fmt) {
        document.getElementById('conv-fmt').value = fmt;
        document.getElementById('selected-fmt-display').innerText = fmt.toUpperCase();
        closeFmtModal();
    }
    
    // Close modal if clicked outside
    window.onclick = function(event) {
        if (event.target == document.getElementById('fmt-modal')) {
            closeFmtModal();
        }
    }

    async function updateLibs() {
        if(confirm("Update libraries (yt-dlp, instaloader)? This may take a moment.")) {
             const res = await window.pywebview.api.update_libraries();
             alert(res.message);
        }
    }
    
    let dlData = null;
    async function dl_analyze() {
        const url = document.getElementById('url').value;
        if(!url) return;
        document.getElementById('dl-status').innerText = "Analyzing...";
        document.getElementById('dl-playlist-info').style.display = 'none';
        try {
            const res = await window.pywebview.api.analyze(url);
            dlData = JSON.parse(res);
            
            // Check if it's a playlist
            if(dlData._type === 'playlist' && dlData.entries) {
                document.getElementById('dl-title').innerText = dlData.title || "Playlist";
                document.getElementById('dl-author').innerText = dlData.uploader || dlData.channel || "";
                document.getElementById('dl-thumb').src = dlData.thumbnail || (dlData.entries[0] && dlData.entries[0].thumbnail) || "";
                
                // Show playlist info
                document.getElementById('pl-title').innerText = dlData.title || "Playlist";
                document.getElementById('pl-count').innerText = `${dlData.entries.length} tracks`;
                document.getElementById('dl-playlist-info').style.display = 'block';
            } else {
                document.getElementById('dl-title').innerText = dlData.title || "Video";
                document.getElementById('dl-author').innerText = dlData.uploader || "";
                document.getElementById('dl-thumb').src = dlData.thumbnail || "";
            }
            
            document.getElementById('dl-card').style.display = 'block';
            document.getElementById('dl-status').innerText = "Ready";
        } catch(e) { document.getElementById('dl-status').innerText = "Error: "+e; }
    }
    function updateDlOptions() {
        const type = document.getElementById('dl-type').value;
        document.getElementById('dl-opt-video').style.display = (type === 'video') ? 'flex' : 'none';
        document.getElementById('dl-opt-audio').style.display = (type === 'audio') ? 'flex' : 'none';
    }
    
    function updateAudioFormat() {
        const audioFmt = document.getElementById('dl-audio-fmt').value;
        const bitrateContainer = document.getElementById('dl-bitrate-container');
        // Hide bitrate for WAV (lossless), show for MP3
        bitrateContainer.style.display = (audioFmt === 'wav') ? 'none' : 'block';
    }

    async function selectCoverArt() {
        const f = await window.pywebview.api.choose_files(false);
        if(f && f.length) {
            const path = f[0];
            document.getElementById('dl-cover-path').value = path;
            document.getElementById('cover-preview').src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(path);
            document.getElementById('cover-preview').style.display = 'block';
            document.getElementById('cover-placeholder').style.display = 'none';
        }
    }
    
    function clearCoverArt() {
        document.getElementById('dl-cover-path').value = "";
        document.getElementById('cover-preview').src = "";
        document.getElementById('cover-preview').style.display = 'none';
        document.getElementById('cover-placeholder').style.display = 'flex';
    }

    async function dl_start() {
        if(!dlData) return;
        
        const type = document.getElementById('dl-type').value;
        const opts = {
            type: type,
            res: document.getElementById('dl-res').value,
            fps: document.getElementById('dl-fps').value,
            bitrate: document.getElementById('dl-bitrate').value,
            audio_fmt: document.getElementById('dl-audio-fmt') ? document.getElementById('dl-audio-fmt').value : 'mp3',
            cover_art: document.getElementById('dl-cover-path') ? document.getElementById('dl-cover-path').value : null,
            is_playlist: dlData._type === 'playlist'
        };

        document.getElementById('dl-status').innerText = "Downloading...";
        document.getElementById('dl-track-progress').style.display = 'none';
        
        const res = await window.pywebview.api.download(dlData.webpage_url || dlData.url, JSON.stringify(opts));
        
        if(res.success) {
            document.getElementById('dl-status').innerText = res.message || "Done!";
            document.getElementById('dl-track-progress').style.display = 'none';
        } else {
            document.getElementById('dl-status').innerText = "Error: " + res.error;
        }
    }

    let convFiles = [];
    async function conv_add() {
        const f = await window.pywebview.api.choose_files(true);
        if(f) { convFiles = [...convFiles, ...f]; conv_render(); }
    }
    function conv_clear() { convFiles = []; conv_render(); }
    function conv_render() {
        const list = document.getElementById('conv-list');
        if(convFiles.length === 0) {
            list.innerHTML = `<div style="text-align:center; color:#555; padding:20px;">No files added yet.</div>`;
            return;
        }
        list.innerHTML = convFiles.map(x => 
            `<div style="background:#222; padding:12px; margin-bottom:8px; border-radius:8px; display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:0.9em; overflow:hidden; text-overflow:ellipsis;">${x.split(/[\\\\/]/).pop()}</span>
                <span style="font-size:0.8em; color:#888; background:#333; padding:2px 6px; border-radius:4px;">Pending</span>
             </div>`
        ).join('');
    }
    async function conv_start() {
        if(!convFiles.length) { alert("Add files first!"); return; }
        const fmt = document.getElementById('conv-fmt').value.trim();
        if(!fmt) { alert("Please select a target format!"); return; }
        
        const folder = await window.pywebview.api.choose_folder();
        if(!folder) return;
        
        document.getElementById('conv-status').innerText = "Starting Batch Conversion...";
        let errors = 0;
        let done = 0;
        
        // Disable buttons
        const btns = document.querySelectorAll('#converter button');
        btns.forEach(b => b.disabled = true);
        
        for(let i=0; i<convFiles.length; i++) {
             const f = convFiles[i];
             // Update status for this file
             document.getElementById('conv-status').innerText = `Converting (${i+1}/${convFiles.length}): ${f.split(/[\\\\/]/).pop()}...`;
             
             const res = await window.pywebview.api.convert(f, fmt, folder);
             if(!res.success) errors++;
             else done++;
        }
        
        btns.forEach(b => b.disabled = false);
        document.getElementById('conv-status').innerText = `Batch Finished! Success: ${done}, Errors: ${errors}`;
        
        // Open folder
        await window.pywebview.api.open_directory(folder);
    }

    let editFile = null;
    let natW=0, natH=0;
    
    async function edit_file() {
        const f = await window.pywebview.api.choose_files(false);
        if(f && f.length) {
            editFile = f[0];
            const url = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(editFile);
            
            const img = document.getElementById('edit-img');
            const vid = document.getElementById('edit-vid');
            img.style.display='none'; vid.style.display='none';
            
            if(editFile.match(/\\.(mp4|avi|mov|mkv|webm)$/i)) {
                vid.src = url;
                vid.style.display = 'block';
                vid.onloadedmetadata = () => { natW=vid.videoWidth; natH=vid.videoHeight; initInputs(); };
            } else {
                img.src = url;
                img.style.display = 'block';
                img.onload = () => { natW=img.naturalWidth; natH=img.naturalHeight; initInputs(); };
            }
        }
    }
    
    function initInputs() {
        document.getElementById('rs-w').value = natW;
        document.getElementById('rs-h').value = natH;
        updateCropInputs(0,0,natW,natH);
    }
    
    let isCrop = false, isDrag = false, resizeDir = null;
    let startX, startY, startLeft, startTop, startW, startH;
    const box = document.getElementById('crop-box');
    
    function toggleCrop() {
        isCrop = !isCrop;
        box.style.display = isCrop ? 'block' : 'none';
        if(isCrop) {
            const p = box.offsetParent;
            box.style.left = '0'; box.style.top = '0';
            box.style.width = p.clientWidth + 'px';
            box.style.height = p.clientHeight + 'px';
            updateBackendCrop();
        }
    }
    
    function updateShape() {
        const shape = document.getElementById('crop-shape').value;
        box.style.borderRadius = (shape === 'circle') ? '50%' : '0';
    }

    window.cropStartDrag = (e) => {
        if(e.target !== box) return; 
        isDrag = true; resizeDir = null;
        startX = e.clientX; startY = e.clientY;
        startLeft = box.offsetLeft; startTop = box.offsetTop;
        document.addEventListener('mousemove', cropOnMove);
        document.addEventListener('mouseup', cropEnd);
    };
    
    window.cropStartResize = (e, dir) => {
        e.stopPropagation();
        isDrag = true; resizeDir = dir;
        startX = e.clientX; startY = e.clientY;
        startW = box.offsetWidth; startH = box.offsetHeight;
        startLeft = box.offsetLeft; startTop = box.offsetTop;
        document.addEventListener('mousemove', cropOnMove);
        document.addEventListener('mouseup', cropEnd);
    };

    function cropOnMove(e) {
        if(!isDrag) return;
        const p = box.offsetParent; 
        const dx = e.clientX - startX;
        const dy = e.clientY - startY;

        if(!resizeDir) { 
            let nl = startLeft + dx, nt = startTop + dy;
            nl = Math.max(0, Math.min(nl, p.clientWidth - box.offsetWidth));
            nt = Math.max(0, Math.min(nt, p.clientHeight - box.offsetHeight));
            box.style.left = nl+'px'; box.style.top = nt+'px';
        } else { 
            let nw = startW, nh = startH, nl = startLeft, nt = startTop;
            
            if(resizeDir.includes('e')) nw = startW + dx;
            if(resizeDir.includes('s')) nh = startH + dy;
            if(resizeDir.includes('w')) { nw = startW - dx; nl = startLeft + dx; }
            if(resizeDir.includes('n')) { nh = startH - dy; nt = startTop + dy; }
            
            if(nw < 20) nw=20; if(nh < 20) nh=20;
            
            if(nl < 0) { nw += nl; nl = 0; } 
            if(nt < 0) { nh += nt; nt = 0; }
            if(nl+nw > p.clientWidth) nw = p.clientWidth - nl;
            if(nt+nh > p.clientHeight) nh = p.clientHeight - nt;

            box.style.width = nw + 'px'; box.style.height = nh + 'px';
            box.style.left = nl + 'px'; box.style.top = nt + 'px';
        }
        updateBackendCrop();
    }
    
    function cropEnd() {
        isDrag = false; resizeDir = null;
        document.removeEventListener('mousemove', cropOnMove);
        document.removeEventListener('mouseup', cropEnd);
    }
    
    function updateBackendCrop() {
        const p = box.offsetParent;
        const media = document.querySelector('#editor-view .media-preview[style*="block"]');
        if(!media) return;
        
        const scaleX = natW / p.clientWidth;
        const scaleY = natH / p.clientHeight; 
        
        const rRatio = natW / natH;
        const cRatio = p.clientWidth / p.clientHeight;
        
        let renderW, renderH, offX=0, offY=0;
        
        if(rRatio > cRatio) { 
            renderW = p.clientWidth;
            renderH = renderW / rRatio;
            offY = (p.clientHeight - renderH) / 2;
        } else { 
            renderH = p.clientHeight;
            renderW = renderH * rRatio;
            offX = (p.clientWidth - renderW) / 2;
        }
        
        let bx = box.offsetLeft - offX;
        let by = box.offsetTop - offY;
        let bw = box.offsetWidth;
        let bh = box.offsetHeight;
        
        const finalX = Math.max(0, Math.round(bx * (natW / renderW)));
        const finalY = Math.max(0, Math.round(by * (natH / renderH)));
        const finalW = Math.round(bw * (natW / renderW));
        const finalH = Math.round(bh * (natH / renderH));
        
        updateCropInputs(finalX, finalY, finalW, finalH);
    }
    
    function updateCropInputs(x,y,w,h) {
        document.getElementById('cp-x').value = x;
        document.getElementById('cp-y').value = y;
        document.getElementById('cp-w').value = w;
        document.getElementById('cp-h').value = h;
    }

    async function edit_process() {
        if(!editFile) return;
        const folder = await window.pywebview.api.choose_folder();
        if(!folder) return;
        
        document.getElementById('edit-status').innerText = "Processing...";
        const args = {
            src: editFile,
            folder: folder,
            w: document.getElementById('rs-w').value,
            h: document.getElementById('rs-h').value,
            crop: isCrop ? {
                x: document.getElementById('cp-x').value,
                y: document.getElementById('cp-y').value,
                w: document.getElementById('cp-w').value,
                h: document.getElementById('cp-h').value,
                shape: document.getElementById('crop-shape').value
            } : null
        };
        const res = await window.pywebview.api.edit_media(JSON.stringify(args));
        document.getElementById('edit-status').innerText = res.success ? "Saved!" : "Error: "+res.error;
    }

    let gifV = null, natGw=0, natGh=0;
    
    let isGifCrop = false, isGifDrag = false, gifResizeDir = null;
    let gx, gy, gl, gt, gw, gh;
    const gBox = document.getElementById('gif-crop-box');
    
    function toggleGifCrop() {
        isGifCrop = !isGifCrop;
        gBox.style.display = isGifCrop ? 'block' : 'none';
        if(isGifCrop && gifV) {
             const p = gBox.offsetParent;
             gBox.style.left='0'; gBox.style.top='0';
             gBox.style.width= p.clientWidth + 'px';
             gBox.style.height= p.clientHeight + 'px';
             updateGifBackendCrop();
        }
    }
    
    window.gifCropStartDrag = (e) => {
        if(e.target !== gBox) return;
        isGifDrag=true; gifResizeDir=null;
        gx=e.clientX; gy=e.clientY; gl=gBox.offsetLeft; gt=gBox.offsetTop;
        document.addEventListener('mousemove', gifOnMove);
        document.addEventListener('mouseup', gifEnd);
    };
    
    window.gifCropStartResize = (e, dir) => {
        e.stopPropagation();
        isGifDrag=true; gifResizeDir=dir;
        gx=e.clientX; gy=e.clientY; gw=gBox.offsetWidth; gh=gBox.offsetHeight; gl=gBox.offsetLeft; gt=gBox.offsetTop;
        document.addEventListener('mousemove', gifOnMove);
        document.addEventListener('mouseup', gifEnd);
    };
    
    function gifOnMove(e) {
        if(!isGifDrag) return;
        const p = gBox.offsetParent; 
        const dx = e.clientX - gx; const dy = e.clientY - gy;
        
        if(!gifResizeDir) {
             let nl = gl + dx, nt = gt + dy;
             nl = Math.max(0, Math.min(nl, p.clientWidth - gBox.offsetWidth));
             nt = Math.max(0, Math.min(nt, p.clientHeight - gBox.offsetHeight));
             gBox.style.left=nl+'px'; gBox.style.top=nt+'px';
        } else {
             let nw=gw, nh=gh, nl=gl, nt=gt;
             if(gifResizeDir.includes('e')) nw = gw + dx;
             if(gifResizeDir.includes('s')) nh = gh + dy;
             if(gifResizeDir.includes('w')) { nw = gw - dx; nl = gl + dx; }
             if(gifResizeDir.includes('n')) { nh = gh - dy; nt = gt + dy; }
             
             if(nw<20) nw=20; if(nh<20) nh=20;
             if(nl<0) { nw+=nl; nl=0; } if(nt<0) { nh+=nt; nt=0; }
             if(nl+nw > p.clientWidth) nw = p.clientWidth - nl;
             if(nt+nh > p.clientHeight) nh = p.clientHeight - nt;
             
             gBox.style.width=nw+'px'; gBox.style.height=nh+'px';
             gBox.style.left=nl+'px'; gBox.style.top=nt+'px';
        }
        updateGifBackendCrop();
    }
    
    function gifEnd() { isGifDrag=false; document.removeEventListener('mousemove', gifOnMove); document.removeEventListener('mouseup', gifEnd); }
    
    function updateGifBackendCrop() {
        const p = gBox.offsetParent;
        const rRatio = natGw / natGh;
        const cRatio = p.clientWidth / p.clientHeight;
        let renderW, renderH, offX=0, offY=0;
        
        if(rRatio > cRatio) { renderW = p.clientWidth; renderH = renderW / rRatio; offY = (p.clientHeight - renderH)/2; }
        else { renderH = p.clientHeight; renderW = renderH * rRatio; offX = (p.clientWidth - renderW)/2; }
        
        let bx = gBox.offsetLeft - offX; let by = gBox.offsetTop - offY;
        
        const finalX = Math.max(0, Math.round(bx * (natGw / renderW)));
        const finalY = Math.max(0, Math.round(by * (natGh / renderH)));
        const finalW = Math.round(gBox.offsetWidth * (natGw / renderW));
        const finalH = Math.round(gBox.offsetHeight * (natGh / renderH));
        
        document.getElementById('gif-crop').value = `${finalX}:${finalY}:${finalW}:${finalH}`;
    }

    async function gif_file() {
        const f = await window.pywebview.api.choose_files(false);
        if(f && f.length) {
            gifV = f[0];
            document.getElementById('gif-ph').style.display='none';
            document.getElementById('gif-preview').style.display='none';
            
            const v = document.getElementById('gif-vid');
            v.src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(gifV);
            v.style.display='block';
            v.onloadedmetadata = () => { natGw=v.videoWidth; natGh=v.videoHeight; };
        }
    }
    async function gif_create() {
        if(!gifV) return;
        const folder = await window.pywebview.api.choose_folder();
        if(!folder) return;
        document.getElementById('gif-status').innerText = "Generating GIF...";
        const conf = {
            src: gifV, folder: folder,
            start: document.getElementById('gif-start').value,
            end: document.getElementById('gif-end').value,
            fps: document.getElementById('gif-fps').value,
            width: document.getElementById('gif-width').value,
            crop: document.getElementById('gif-crop').value 
        };
        const res = await window.pywebview.api.make_gif(JSON.stringify(conf));
        if(res.success) {
            document.getElementById('gif-status').innerText = "GIF Created!";
            document.getElementById('gif-vid').style.display='none';
            const img = document.getElementById('gif-preview');
            img.src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(res.path) + "&t=" + new Date().getTime();
            img.style.display='block';
        } else {
             document.getElementById('gif-status').innerText = "Error: " + res.error;
        }
    }
    
    async function shorten_url() {
        const url = document.getElementById('short-url-input').value.trim();
        const alias = document.getElementById('short-alias').value.trim();
        if(!url) return;
        
        document.getElementById('short-status').innerText = "Shortening...";
        document.getElementById('short-result').style.display = 'none';
        
        const res = await window.pywebview.api.shorten_url(url, alias);
        if(res.success) {
             document.getElementById('short-url-display').innerText = res.url;
             document.getElementById('short-result').style.display = 'block';
             document.getElementById('short-status').innerText = "";
             
             // Auto-fill QR and Scroll
             document.getElementById('avr-data').value = res.url;
             debounceAvr();
             document.getElementById('qr-section-inner').scrollIntoView({behavior:'smooth'});
        } else {
             document.getElementById('short-status').innerText = "Error: " + res.error;
        }
    }

    // --- ADVANCED QR JS ---
    let avrDebounce;
    let avrLogoPath = null;
    let avrFinalPath = null;

    function switchQrTab(t) {
        document.querySelectorAll('.config-tab').forEach(e => e.classList.remove('active'));
        document.querySelectorAll('.qr-config .nav-item').forEach(e => e.classList.remove('active'));
        
        document.getElementById('qt-'+t).classList.add('active');
        const tabs = ['content','shapes','colors','logo'];
        const idx = tabs.indexOf(t);
        if(idx>=0) document.querySelectorAll('.qr-config .nav-item')[idx].classList.add('active');
    }
    
    function selShape(el, shape) {
        document.querySelectorAll('.shape-item').forEach(e => e.classList.remove('selected'));
        el.classList.add('selected');
        document.getElementById('avr-shape').value = shape;
        debounceAvr();
    }
    
    function switchColorMode(m) {
        document.getElementById('avr-color-mode').value = m;
        document.getElementById('cm-btn-solid').className = m === 'solid' ? 'color-opt active' : 'color-opt';
        document.getElementById('cm-btn-gradient').className = m === 'gradient' ? 'color-opt active' : 'color-opt';
        document.getElementById('cm-panel-solid').style.display = m === 'solid' ? 'block' : 'none';
        document.getElementById('cm-panel-gradient').style.display = m === 'gradient' ? 'block' : 'none';
        debounceAvr();
    }

    function debounceAvr() {
        clearTimeout(avrDebounce);
        document.getElementById('avr-loading').style.display = 'block';
        avrDebounce = setTimeout(generateAvrQr, 800);
    }
    
    async function generateAvrQr() {
        const data = document.getElementById('avr-data').value;
        if(!data) {
             document.getElementById('avr-preview').src = "";
             document.getElementById('avr-loading').style.display = 'none';
             return;
        }
        
        const mode = document.getElementById('avr-color-mode').value;
        
        const args = {
            data: data,
            drawer: document.getElementById('avr-shape').value,
            fill_color: document.getElementById('avr-fill').value,
            back_color: document.getElementById('avr-back').value,
            logo_path: avrLogoPath,
            mask: mode === 'solid' ? 'solid' : document.getElementById('avr-grad-type').value
        };
        
        if(mode === 'gradient') {
            args.gradient_center = document.getElementById('avr-g-start').value;
            args.gradient_edge = document.getElementById('avr-g-end').value;
            args.gradient_start = document.getElementById('avr-g-start').value;
            args.gradient_end = document.getElementById('avr-g-end').value;
        }
        
        try {
            const res = await window.pywebview.api.generate_advanced_qr(JSON.stringify(args));
            if(res.success) {
                avrFinalPath = res.path;
                document.getElementById('avr-preview').src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(res.path) + "&t=" + new Date().getTime();
            } else {
                console.error(res.error);
            }
        } catch(e) { console.error(e); }
        document.getElementById('avr-loading').style.display = 'none';
    }
    
    async function uploadAvrLogo() {
        const f = await window.pywebview.api.choose_files(false);
        if(f && f.length) {
            avrLogoPath = f[0];
            document.getElementById('avr-logo-name').innerText = f[0].split(/[\\\\/]/).pop();
            document.getElementById('avr-rm-logo').style.display = 'block';
            debounceAvr();
        }
    }
    
    function rmAvrLogo(e) {
        e.stopPropagation();
        avrLogoPath = null;
        document.getElementById('avr-logo-name').innerText = "Click to Upload";
        document.getElementById('avr-rm-logo').style.display = 'none';
        debounceAvr();
    }
    
    async function downloadAvr() {
        if(!avrFinalPath) return;
        const folder = await window.pywebview.api.choose_folder();
        if(folder) {
            const res = await window.pywebview.api.save_qr_cleanup(avrFinalPath, folder);
            if(res.success) alert("Saved!");
        }
    }
    
    function copyShortUrl() {
        const text = document.getElementById("short-url-display").innerText;
        navigator.clipboard.writeText(text);
        
        const btn = document.querySelector('.btn-copy');
        const originalHtml = btn.innerHTML;
        btn.innerHTML = "Copied!";
        btn.style.background = "#44ff44";
        setTimeout(() => {
            btn.innerHTML = originalHtml;
            btn.style.background = "";
        }, 2000);
    }

    // --- BACKGROUND REMOVER ---
    let bgFile = null;
    let bgResultPath = null;
    let bgSessionId = null;
    let bgTool = 'restore'; // 'restore' or 'erase'
    let bgIsDrawing = false;
    let bgStrokePoints = [];
    
    async function bg_open() {
        const f = await window.pywebview.api.choose_files(false);
        if(f && f.length) {
            bgFile = f[0];
            document.getElementById('bg-ph').style.display = 'none';
            document.getElementById('bg-result').style.display = 'none';
            document.getElementById('bg-preview').style.display = 'block';
            document.getElementById('bg-preview').src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(bgFile);
            
            document.getElementById('btn-bg-process').disabled = false;
            document.getElementById('btn-bg-save').disabled = true;
            document.getElementById('bg-tools').style.display = 'none';
            document.getElementById('bg-canvas').style.display = 'none';
            document.getElementById('bg-status').innerText = "Ready to process.";
        }
    }
    
    async function bg_process() {
        if(!bgFile) return;
        
        const model = document.getElementById('bg-model').value;
        const alpha = document.getElementById('bg-alpha').checked; // Kept for future compatibility
        
        document.getElementById('bg-status').innerText = "Processing... (Loading " + model + "...)";
        document.getElementById('btn-bg-process').disabled = true;
        
        // Pass mode='remove_bg' explicitly
        const res = await window.pywebview.api.remove_bg(bgFile, model, 'remove_bg');
        
        if(res.success) {
            bgResultPath = res.path;
            bgSessionId = res.session_id; // Capture session ID
            
            const timeToken = new Date().getTime();
            document.getElementById('bg-result').src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(res.path) + "&t=" + timeToken;
            document.getElementById('bg-preview').style.display = 'none'; 
            document.getElementById('bg-result').style.display = 'block';
            
            // Enable Tools
            document.getElementById('bg-tools').style.display = 'block';
            document.getElementById('bg-canvas').style.display = 'block';
            initBgCanvas();
            
            document.getElementById('bg-status').innerText = "Background Removed! Use tools to refine.";
            document.getElementById('btn-bg-save').disabled = false;
        } else {
             document.getElementById('bg-status').innerText = "Error: " + res.error;
             document.getElementById('btn-bg-process').disabled = false;
        }
    }
    
    // Brush Tool Logic
    function setBgTool(mode) {
        bgTool = mode;
        document.getElementById('btn-tool-brush').className = mode === 'restore' ? 'btn btn-sm active' : 'btn btn-sm';
        document.getElementById('btn-tool-erase').className = mode === 'erase' ? 'btn btn-sm active' : 'btn btn-sm';
        
        // Visual feedback
        const btnB = document.getElementById('btn-tool-brush');
        const btnE = document.getElementById('btn-tool-erase');
        btnB.style.background = mode === 'restore' ? '#444' : '#222';
        btnE.style.background = mode === 'erase' ? '#444' : '#222';
    }
    
    function initBgCanvas() {
        const c = document.getElementById('bg-canvas');
        const img = document.getElementById('bg-result');
        const container = document.querySelector('#bgremover .editor-container');
        
        // Hàm tính toán vị trí và kích thước canvas khớp với ảnh
        const updateCanvasSize = () => {
            const w = img.naturalWidth;
            const h = img.naturalHeight;
            const cw = container.clientWidth;
            const ch = container.clientHeight;
            
            if(!w || !h) return;
            
            // Tính scale để fit ảnh vào container (object-fit: contain)
            const scale = Math.min(cw / w, ch / h);
            const displayW = w * scale;
            const displayH = h * scale;
            
            // Tính offset để center ảnh
            const left = (cw - displayW) / 2;
            const top = (ch - displayH) / 2;
            
            // Set canvas position và size khớp với ảnh hiển thị
            c.style.position = 'absolute';
            c.style.left = left + 'px';
            c.style.top = top + 'px';
            c.width = displayW;
            c.height = displayH;
            c.style.width = displayW + 'px';
            c.style.height = displayH + 'px';
            
            // Lưu thông tin để dùng sau
            c.dataset.displayWidth = displayW;
            c.dataset.displayHeight = displayH;
            c.dataset.naturalWidth = w;
            c.dataset.naturalHeight = h;
        };
        
        // Chạy khi ảnh load xong
        if(img.complete && img.naturalWidth > 0) {
            updateCanvasSize();
        }
        img.addEventListener('load', updateCanvasSize);
        
        // Theo dõi resize
        const observer = new ResizeObserver(updateCanvasSize);
        observer.observe(container);
        
        // Clear canvas
        const ctx = c.getContext('2d');
        ctx.clearRect(0, 0, c.width, c.height);
        
        // Chỉ bind events một lần
        if(c.dataset.bound) return;
        c.dataset.bound = 'true';
        
        // Mouse events
        c.addEventListener('mousedown', startDraw);
        c.addEventListener('mousemove', draw);
        c.addEventListener('mouseup', endDraw);
        c.addEventListener('mouseleave', endDraw);
        
        // Touch events cho mobile
        c.addEventListener('touchstart', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousedown', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            c.dispatchEvent(mouseEvent);
        });
        
        c.addEventListener('touchmove', (e) => {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent('mousemove', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            c.dispatchEvent(mouseEvent);
        });
        
        c.addEventListener('touchend', (e) => {
            e.preventDefault();
            const mouseEvent = new MouseEvent('mouseup', {});
            c.dispatchEvent(mouseEvent);
        });
        
        // Brush size slider
        document.getElementById('bg-brush-size').addEventListener('input', (e) => {
            document.getElementById('bg-brush-val').innerText = e.target.value;
        });
    }
    
    function startDraw(e) {
        if(!bgSessionId) return;
        
        bgIsDrawing = true;
        bgStrokePoints = [];
        
        // Vẽ điểm đầu tiên
        const c = document.getElementById('bg-canvas');
        const rect = c.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        bgStrokePoints.push([Math.round(x), Math.round(y)]);
        
        // Vẽ một chấm tròn tại vị trí click
        const ctx = c.getContext('2d');
        const size = parseInt(document.getElementById('bg-brush-size').value);
        
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.globalAlpha = 0.5;
        
        ctx.fillStyle = bgTool === 'restore' ? 'rgba(0, 255, 100, 1)' : 'rgba(255, 50, 50, 1)';
        ctx.beginPath();
        ctx.arc(x, y, size / 2, 0, Math.PI * 2);
        ctx.fill();
    }
    
    function draw(e) {
        if(!bgIsDrawing) return;
        
        const c = document.getElementById('bg-canvas');
        const rect = c.getBoundingClientRect();
        
        // Tọa độ tương đối canvas
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        // Kiểm tra trong bounds
        if(x < 0 || y < 0 || x > c.width || y > c.height) return;
        
        const lastPoint = bgStrokePoints[bgStrokePoints.length - 1];
        const distance = lastPoint ? Math.sqrt(Math.pow(x - lastPoint[0], 2) + Math.pow(y - lastPoint[1], 2)) : 0;
        
        // Chỉ thêm point nếu di chuyển đủ xa (tránh spam)
        if(distance > 2 || bgStrokePoints.length === 0) {
            bgStrokePoints.push([Math.round(x), Math.round(y)]);
        } else {
            return; // Không vẽ nếu quá gần điểm trước
        }
        
        const ctx = c.getContext('2d');
        const size = parseInt(document.getElementById('bg-brush-size').value);
        
        ctx.globalAlpha = 0.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = size;
        ctx.strokeStyle = bgTool === 'restore' ? 'rgba(0, 255, 100, 1)' : 'rgba(255, 50, 50, 1)';
        
        // Vẽ đường nối
        if(bgStrokePoints.length >= 2) {
            const prev = bgStrokePoints[bgStrokePoints.length - 2];
            ctx.beginPath();
            ctx.moveTo(prev[0], prev[1]);
            ctx.lineTo(x, y);
            ctx.stroke();
        }
    }
    
    async function endDraw() {
        if(!bgIsDrawing) return;
        bgIsDrawing = false;
        
        if(bgStrokePoints.length === 0) return;
        
        const c = document.getElementById('bg-canvas');
        const size = parseInt(document.getElementById('bg-brush-size').value);
        
        // Lấy kích thước display và natural từ dataset
        const displayW = parseFloat(c.dataset.displayWidth);
        const displayH = parseFloat(c.dataset.displayHeight);
        
        if(!displayW || !displayH) {
            console.error('Canvas size not initialized');
            return;
        }
        
        // Gửi stroke về backend
        const stroke = {
            points: bgStrokePoints,
            mode: bgTool, // 'restore' hoặc 'erase'
            radius: size
        };
        
        const displaySize = [Math.round(displayW), Math.round(displayH)];
        
        document.getElementById('bg-status').innerText = "Applying stroke...";
        
        try {
            const res = await window.pywebview.api.bg_edit(bgSessionId, [stroke], displaySize);
            
            if(res.success) {
                // Cập nhật preview
                if(res.preview) {
                    const img = document.getElementById('bg-result');
                    if(res.preview.startsWith('data:')) {
                        img.src = res.preview;
                    } else {
                        img.src = "data:image/png;base64," + res.preview;
                    }
                    
                    // Chờ ảnh load xong rồi mới clear canvas
                    img.onload = () => {
                        const ctx = c.getContext('2d');
                        ctx.clearRect(0, 0, c.width, c.height);
                    };
                } else {
                    // Nếu không có preview, vẫn GIỮ canvas để user thấy được nét vẽ
                    console.warn('No preview returned from server');
                }
                
                document.getElementById('bg-status').innerText = "Stroke applied successfully.";
            } else {
                document.getElementById('bg-status').innerText = "Failed: " + (res.error || "Unknown error");
                // Vẫn giữ canvas khi lỗi để user thấy được họ đã vẽ gì
            }
        } catch(error) {
            console.error('Edit failed:', error);
            document.getElementById('bg-status').innerText = "Error applying stroke";
            // Vẫn giữ canvas khi lỗi
        }
        
        // KHÔNG clear canvas ở đây nữa - chỉ clear khi có ảnh mới load xong
        // Điều này cho phép user thấy nét vẽ của mình trong khi chờ server xử lý
        
        // Reset stroke points
        bgStrokePoints = [];
    }
    
    async function bg_undo() {
        if(!bgSessionId) return;
        const res = await window.pywebview.api.bg_undo(bgSessionId);
        handleEditRes(res, "Undo");
    }
    
    async function bg_redo() {
        if(!bgSessionId) return;
        const res = await window.pywebview.api.bg_redo(bgSessionId);
        handleEditRes(res, "Redo");
    }
    
    function handleEditRes(res, action) {
        if(res.success) {
            if(res.preview) { // Check if we have a preview
                 if(res.preview.startsWith('data:')) {
                     document.getElementById('bg-result').src = res.preview;
                 } else {
                     document.getElementById('bg-result').src = "data:image/png;base64," + res.preview;
                 }
            }
            document.getElementById('bg-status').innerText = action + " successful.";
        } else {
            document.getElementById('bg-status').innerText = action + " failed: " + res.error;
        }
    }
    
    async function bg_save() {
        if(!bgResultPath && !bgSessionId) return;
        
        const folder = await window.pywebview.api.choose_folder();
        if(folder) {
             if(bgSessionId) {
                 const res = await window.pywebview.api.bg_save_export(bgSessionId, folder);
                 if(res.success) alert("Saved to " + res.path);
                 else alert("Save failed: " + res.error);
             } else {
                 // Fallback
                 const res = await window.pywebview.api.save_qr_cleanup(bgResultPath, folder);
                 if(res.success) alert("Saved!");
             }
        }
    }
    // --- WAVEAUTH START v4.0 ---
    let waFiles = [];
    let waCurrentData = null; 

    async function wa_addFiles() {
        const f = await window.pywebview.api.choose_files(true);
        if(f) {
            waFiles = [...waFiles, ...f];
            wa_updateQ();
        }
    }
    
    function wa_clearQ() {
        waFiles = [];
        wa_updateQ();
    }
    
    function wa_removeQ(idx) {
        waFiles.splice(idx, 1);
        wa_updateQ();
    }
    
    function wa_updateQ() {
        const q = document.getElementById('wa-queue');
        if(waFiles.length === 0) {
            q.innerHTML = '<div style="text-align:center; color:#555; font-size:0.9em;">Queue empty</div>';
            document.getElementById('wa-btn-run').disabled = true;
            return;
        }
        
        q.innerHTML = waFiles.map((f, i) => `
            <div style="display:flex; justify-content:space-between; align-items:center; background:#222; padding:8px; margin-bottom:5px; border-radius:6px; font-size:0.85em;">
                <div style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:180px;">${f.split(/[\\\\/]/).pop()}</div>
                <div style="display:flex; align-items:center; gap:5px;">
                    <span style="color:#888; font-size:0.8em;" id="wa-q-status-${i}">Pending</span>
                    <button style="background:none; border:none; color:#ff5555; cursor:pointer; font-weight:bold;" onclick="wa_removeQ(${i})" title="Remove">×</button>
                </div>
            </div>
        `).join('');
        document.getElementById('wa-btn-run').disabled = false;
    }
    
    async function wa_processQ() {
        if(waFiles.length === 0) return;
        
        const btn = document.getElementById('wa-btn-run');
        const status = document.getElementById('wa-status');
        btn.disabled = true;
        status.innerText = "Starting Batch...";
        
        // Copy queue logic processed one by one
        let processed = 0;
        // Disable remove buttons during processing
        const removeBtns = document.querySelectorAll('#wa-queue button');
        removeBtns.forEach(b => b.disabled = true);
        
        for(let i=0; i<waFiles.length; i++) {
            const f = waFiles[i];
            const statEl = document.getElementById(`wa-q-status-${i}`);
            statEl.style.color = "#00aaff";
            statEl.innerText = "Running...";
            status.innerText = `Analyzing ${i+1}/${waFiles.length}: ${f.split(/[\\\\/]/).pop()}...`;
            
            try {
                const res = await window.pywebview.api.wa_analyze(f);
                if(res.success) {
                    statEl.style.color = "#00ff88";
                    statEl.innerText = "Done";
                    wa_render(res);
                } else {
                    statEl.style.color = "#ff0055";
                    statEl.innerText = "Error";
                    console.error(res.error);
                }
            } catch(e) {
                 statEl.style.color = "#ff0055";
                 statEl.innerText = "Fail";
            }
            
            processed++;
            wa_loadHistory();
        }
        
        // Safe Queue Clearing as requested
        waFiles = [];
        wa_updateQ();
        
        btn.disabled = false;
        status.innerText = `Batch Complete. Processed ${processed} files.`;
    }
    
    function wa_render(data) {
        waCurrentData = data;
        document.getElementById('wa-filename').innerText = data.filename;
        document.getElementById('wa-verdict').innerText = data.verdict;
        document.getElementById('wa-score').innerText = `Score: ${data.score}/100`;
        
        const v = document.getElementById('wa-score');
        if(data.score >= 90) v.style.color = "#00ff88"; 
        else if(data.score >= 70) v.style.color = "#ffaa00"; 
        else v.style.color = "#ff0055"; 
        
        // Detailed Metrics
        document.getElementById('wa-slope').innerText = data.metrics ? data.metrics.slope_metric : "-";
        document.getElementById('wa-bw').innerText = data.metrics ? data.metrics.bandwidth_ratio : "-";
        document.getElementById('wa-dr').innerText = data.metrics ? data.metrics.dynamic_range + " dB" : "-";
        document.getElementById('wa-fmt').innerText = data.format || "-";
        document.getElementById('wa-mp3').innerText = data.mp3_profile || "-";
        
        let cutoffDisplay = "-";
        if(data.cutoff_freq_hz) cutoffDisplay = data.cutoff_freq_hz + " Hz";
        else if(data.cutoff) cutoffDisplay = data.cutoff;
        
        document.getElementById('wa-dur').innerHTML = Math.round(data.duration) + "s <br><span style='font-size:0.8em; color:#888;'>Cutoff: " + cutoffDisplay + "</span>";
        
        // Reasons
        const reasonslist = document.getElementById('wa-reasons');
        if(data.reasons && data.reasons.length > 0) {
            reasonslist.innerHTML = data.reasons.map(r => `<div>• ${r}</div>`).join('');
        } else {
            reasonslist.innerHTML = "";
        }
        
        const img = document.getElementById('wa-spectrogram');
        img.src = "data:image/png;base64," + data.spectrogram;
        img.style.display = 'block';
        document.getElementById('wa-ph').style.display = 'none';
        
        document.getElementById('wa-btn-export').disabled = false;
    }
    
    async function wa_loadHistory() {
        const hist = await window.pywebview.api.wa_history();
        const container = document.getElementById('wa-history');
        
        // Dashboard Stats
        let total = hist.length;
        let avgScore = 0;
        let avgTime = 0;
        
        if(total > 0) {
            avgScore = hist.reduce((sum, h) => sum + (h.score||0), 0) / total;
            const times = hist.map(h => parseFloat((h.process_time||"0").replace('s','')));
            avgTime = times.reduce((sum, t) => sum + t, 0) / total;
        }
        
        document.getElementById('wa-dash-files').innerText = total;
        document.getElementById('wa-dash-score').innerText = avgScore.toFixed(1);
        document.getElementById('wa-dash-time').innerText = avgTime.toFixed(2) + "s";
        
        if(!hist || hist.length === 0) {
            container.innerHTML = '<div style="color:#555; text-align:center; font-size:0.9em; padding:10px;">No history</div>';
            return;
        }
        
        container.innerHTML = hist.map(item => `
            <div style="background:#222; border-bottom:1px solid #333; padding:8px; font-size:0.85em; cursor:pointer;" onclick='wa_render_hist(${JSON.stringify(item).replace(/'/g, "&#39;")})'>
                <div style="display:flex; justify-content:space-between;">
                    <span style="font-weight:bold; color:#ccc; max-width:140px; overflow:hidden; text-overflow:ellipsis;">${item.filename}</span>
                    <span style="${item.score>=90?'color:#00ff88':(item.score>=50?'color:#ffaa00':'color:#ff0055')}">${item.score}/100</span>
                </div>
                <div style="font-size:0.9em; color:#666;">${item.verdict}</div>
            </div>
        `).join('');
    }
    
    function wa_render_hist(item) {
        waCurrentData = null; 
        
        document.getElementById('wa-filename').innerText = item.filename;
        document.getElementById('wa-verdict').innerText = item.verdict;
        document.getElementById('wa-score').innerText = `Score: ${item.score}/100`;
        
        const v = document.getElementById('wa-score');
        if(item.score >= 90) v.style.color = "#00ff88"; 
        else if(item.score >= 70) v.style.color = "#ffaa00"; 
        else v.style.color = "#ff0055";
        
        document.getElementById('wa-slope').innerText = item.metrics ? item.metrics.slope_metric : "-";
        document.getElementById('wa-bw').innerText = item.metrics ? item.metrics.bandwidth_ratio : "-";
        document.getElementById('wa-dr').innerText = item.metrics ? item.metrics.dynamic_range + " dB" : "-";
        document.getElementById('wa-fmt').innerText = item.format || "-";
        document.getElementById('wa-mp3').innerText = item.mp3_profile || "-";
        
        let cutoffDisplay = "-";
        if(item.cutoff_freq_hz) cutoffDisplay = item.cutoff_freq_hz + " Hz";
        else if(item.cutoff) cutoffDisplay = item.cutoff;
        
        document.getElementById('wa-dur').innerHTML = Math.round(item.duration) + "s <br><span style='font-size:0.8em; color:#888;'>Cutoff: " + cutoffDisplay + "</span>";

        const reasonslist = document.getElementById('wa-reasons');
        if(item.reasons && item.reasons.length > 0) {
            reasonslist.innerHTML = item.reasons.map(r => `<div>• ${r}</div>`).join('');
        } else {
             reasonslist.innerHTML = "";
        }
        
        const img = document.getElementById('wa-spectrogram');
        const ph = document.getElementById('wa-ph');
        
        // Safety Clear
        img.src = ""; 
        img.style.display = 'none';
        
        if (item.temp_image) {
             const timeToken = new Date().getTime(); // Anti-cache
             img.src = "http://127.0.0.1:8000/stream?path=" + encodeURIComponent(item.temp_image) + "&t=" + timeToken;
             img.style.display = 'block';
             ph.style.display = 'none';
        } else {
            // Explicitly ensure it's hidden and src is empty
            img.style.display = 'none';
            ph.style.display = 'block';
            ph.innerText = "Spectrogram not saved in history";
        }
        
        document.getElementById('wa-btn-export').disabled = true;
    }

    async function wa_clearHistory() {
        await window.pywebview.api.wa_clear();
        wa_loadHistory();
    }
    
    function wa_showRef() {
        document.getElementById('wa-ref-modal').style.display = 'block';
    }
    
    function wa_exportImg() {
        if(!waCurrentData || !waCurrentData.spectrogram) return;
        const a = document.createElement('a');
        a.href = "data:image/png;base64," + waCurrentData.spectrogram;
        a.download = `spectrogram_${waCurrentData.filename}.png`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    async function wa_exportCSV() {
        const csvContent = await window.pywebview.api.wa_export_csv();
        if(!csvContent) { alert("No history to export"); return; }
        
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `audio_report_${new Date().getTime()}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }
    
    async function wa_clearTempImages() {
        if(!confirm("Delete all temporary analysis images?")) return;
        const res = await window.pywebview.api.wa_clear_temp();
        if(res) alert("Temp images cleared.");
        else alert("Failed to clear temp images.");
    }
    
    // Init History on load
    window.addEventListener('DOMContentLoaded', () => {
        wa_loadHistory();
    });
    // --- WAVEAUTH END ---
</script>
</body>
</html>
"""
