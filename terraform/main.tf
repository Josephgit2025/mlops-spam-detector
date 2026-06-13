# Récupère ma clé SSH enregistrée sur DigitalOcean
data "digitalocean_ssh_key" "ansible-key" {
  name = var.ssh_key_name
}

# Droplet pour Mlflow
resource "digitalocean_droplet" "mlflow" {
    name   = "mlflow-server"
    region = var.region
    image  = "ubuntu-22-04-x64"
    size   = "s-1vcpu-2gb"
    ssh_keys = [data.digitalocean_ssh_key.ansible-key.id]        # autorise ma clé SSH
}

# Cluster DOKS pour l'API
resource "digitalocean_kubernetes_cluster" "spam" {
    name    = "spam-detector-cluster"
    region  = var.region
    version = "1.36.0-do.1"

    node_pool {
        name       = "default-pool"
        size       = "s-2vcpu-2gb"
        node_count = 2
    }
}

resource "digitalocean_database_cluster" "mlflow_db" {
  name       = "mlflow-postgres"
  engine     = "pg"
  version    = "15"
  size       = "db-s-1vcpu-1gb"
  region     = var.region
  node_count = 1
}

output "db_uri" {
  value     = digitalocean_database_cluster.mlflow_db.uri
  sensitive = true
}

resource "digitalocean_firewall" "mlflow" {
  name        = "mlflow-firewall"
  droplet_ids = [digitalocean_droplet.mlflow.id]

  inbound_rule {
    protocol         = "tcp"
    port_range       = "22"
    source_addresses = ["0.0.0.0/0"]
  }

  inbound_rule {
    protocol         = "tcp"
    port_range       = "5000"
    source_addresses = ["0.0.0.0/0"]
  }

  outbound_rule {
    protocol              = "tcp"
    port_range            = "all"
    destination_addresses = ["0.0.0.0/0"]
  }

  outbound_rule {
    protocol              = "udp"
    port_range            = "all"
    destination_addresses = ["0.0.0.0/0"]
  }
}


/*
# Créer un bucket (dossier cloud) pour stocker les modèles
# DO spaces pour stocker les modèles et les artefacts
resource "digitalocean_spaces_bucket" "models" {
    name   = "spam-detector-models"
    region = var.region
    acl    = "private"      # accès privé, rend le bucket privé
}
*/