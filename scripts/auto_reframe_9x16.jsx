// scripts/auto_reframe_9x16.jsx
// Script ExtendScript pour Premiere Pro - Auto Reframe

(function() {
    
    var project = app.project;
    
    if (!project) {
        alert("Aucun projet ouvert dans Premiere Pro");
        return;
    }
    
    // Fonction principale
    function autoReframe916() {
        
        var sequence = project.activeSequence;
        if (!sequence) {
            alert("Aucune séquence active");
            return;
        }
        
        // Paramètres 9:16
        var newWidth = 1080;
        var newHeight = 1920;
        
        // Création d'une nouvelle séquence 9:16
        var verticalSeq = project.createNewSequence("Vertical_" + sequence.name, "HDV-720p25");
        
        // Configuration de la séquence
        verticalSeq.videoTracks[0].overwriteClip(sequence.videoTracks[0].clips[0], 0);
        
        // Application de l'effet Auto Reframe (Premiere Pro 2019+)
        try {
            var effect = verticalSeq.videoTracks[0].clips[0].components[1];
            effect.properties[0].setValue(newWidth, true);
            effect.properties[1].setValue(newHeight, true);
            
            alert("Auto Reframe 9:16 appliqué avec succès!");
            
        } catch(e) {
            alert("Erreur lors de l'application de l'Auto Reframe: " + e.toString());
        }
    }
    
    // Exécution
    autoReframe916();
    
})();
