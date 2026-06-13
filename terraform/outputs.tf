output "mlflow_ip" {
    description = "IP du serveur Mlflow"
    value       = digitalocean_droplet.mlflow.ipv4_address
    # afficher l'IP du serveur Mlflow
}

output "cluster_id" {
    description = "ID du cluster DOKS"
    value       = digitalocean_kubernetes_cluster.spam.id
    # afficher l'ID du cluster DOKS
}

/*
output "spaces_bucket" {
    description = "Nom du bucket DO Spaces"
    value       = digitalocean_spaces_bucket.models.name
    # afficher le nom du bucket DO Spaces
}
*/