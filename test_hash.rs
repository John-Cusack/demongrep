use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

fn main() {
    let paths = vec![
        r"C:\MySource\repos\BOIN.Aprimo",
        r"C:\MySource\repos\BOIN.Aprimo\src",
        r"C:\Users\develterf\source\repos\BOIN.Aprimo",
        r"C:\Users\develterf\source\repos\BOIN.Aprimo\src",
        r"C:/MySource/repos/BOIN.Aprimo",
        r"C:/MySource/repos/BOIN.Aprimo/src",
    ];
    
    for path_str in paths {
        let path = PathBuf::from(path_str);
        if let Ok(canonical) = path.canonicalize() {
            let mut hasher = DefaultHasher::new();
            canonical.hash(&mut hasher);
            let hash = hasher.finish();
            println!("{:x} <- {:?} -> {:?}", hash, path_str, canonical);
        } else {
            println!("FAILED <- {:?}", path_str);
        }
    }
    
    println!("\nTarget hash: bb87f995b6adf622");
}
