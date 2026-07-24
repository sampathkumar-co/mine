export interface PersistedUpload {
  key: string;
  uploadId: string;
  projectId: string;
  kind: string;
  file: File;
  updatedAt: number;
}

const DATABASE = "director-os-uploads";
const STORE = "uploads";

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DATABASE, 1);
    request.onupgradeneeded = () => {
      if (!request.result.objectStoreNames.contains(STORE)) {
        request.result.createObjectStore(STORE, { keyPath: "key" });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function transaction<T>(mode: IDBTransactionMode, operation: (store: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  if (typeof indexedDB === "undefined") throw new Error("IndexedDB is unavailable");
  const database = await openDatabase();
  try {
    return await new Promise<T>((resolve, reject) => {
      const tx = database.transaction(STORE, mode);
      const request = operation(tx.objectStore(STORE));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
      tx.onerror = () => reject(tx.error);
    });
  } finally {
    database.close();
  }
}

export function uploadFingerprint(projectId: string, kind: string, file: File): string {
  return [projectId, kind, file.name, file.size, file.lastModified].join(":");
}

export async function loadUpload(key: string): Promise<PersistedUpload | undefined> {
  return transaction("readonly", (store) => store.get(key));
}

export async function saveUpload(upload: PersistedUpload): Promise<void> {
  await transaction("readwrite", (store) => store.put(upload));
}

export async function removeUpload(key: string): Promise<void> {
  await transaction("readwrite", (store) => store.delete(key));
}
