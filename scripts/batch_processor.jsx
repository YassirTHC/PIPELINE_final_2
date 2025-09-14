// BATCH_PROCESSOR.jsx - Traitement par lots
// ================================

// Script pour traiter plusieurs clips automatiquement
(function() {
    
    var inputFolder = Folder.selectDialog("Sélectionnez le dossier des clips");
    var outputFolder = Folder.selectDialog("Sélectionnez le dossier de sortie");
    
    if (!inputFolder || !outputFolder) {
        alert("Dossiers non sélectionnés");
        return;
    }
    
    var files = inputFolder.getFiles("*.mp4");
    
    for (var i = 0; i < files.length; i++) {
        processClip(files[i], outputFolder);
    }
    
    function processClip(clipFile, outputDir) {
        // Import du clip
        var importOptions = new ImportOptions();
        importOptions.file = clipFile;
        var clip = app.project.importFiles([importOptions.file]);
        
        // Création séquence 9:16
        var sequence = app.project.createNewSequence(clipFile.name, "HDV-1080i25");
        sequence.videoTracks[0].insertClip(clip[0], 0);
        
        // Export
        var outputFile = new File(outputDir.fsName + "/" + clipFile.name);
        app.encoder.encodeSequence(sequence, outputFile.fsName, "H.264", false);
    }
    
    alert("Traitement par lots terminé!");
    
})();