pragma solidity ^0.8.28;

/**
 * @title FederatedLearningLog
 * @notice Stores immutable training round records for a Federated Learning system.
 *         Each round logs:
 *           - model hash
 *           - Filecoin/IPFS artifact CID
 *           - participating client addresses
 *           - per-client contributions (samples + score)
 *
 *         The contract does NOT store model weights or training data.
 *         The goal is to create a verifiable, decentralized audit log
 *         that references Filecoin-stored artifacts.
 */
contract FederatedLearningLog {

    /// STRUCTS

    /// @notice High-level record for each training round.
    struct RoundInfo {
        bytes32 modelHash;        // Hash of global model weights (e.g., sha256)
        string artifactCid;       // Filecoin/IPFS CID of the artifact bundle (model file)
        string summaryCid;         // Filecoin/IPFS CID of the summary data (for LLM queries)
        address[] clients;        // Participating client wallet addresses
        uint256 timestamp;        // Block timestamp of the round
    }

    /// @notice Per-client data for each round.
    struct ClientRoundInfo {
        uint256 samples;          // Number of local samples used by the client
        uint256 score;            // Contribution score (computed off-chain)
    }

    /// STATE VARIABLES

    address public owner;         // Admin
    address public server;        // Federated Learning server wallet

    uint256 public latestRoundId; // Tracks last recorded round ID

    mapping(address => bool) public isClient;   // Registered client whitelist
    mapping(uint256 => RoundInfo) public rounds;
    mapping(uint256 => mapping(address => ClientRoundInfo)) public clientRounds;

    /// EVENTS

    event ServerUpdated(address indexed newServer);
    event ClientRegistered(address indexed client);
    event ClientRemoved(address indexed client);

    event RoundRecorded(
        uint256 indexed roundId,
        bytes32 modelHash,
        string artifactCid,
        string summaryCid,
        address[] clients,
        uint256 timestamp
    );

    event ClientContributionRecorded(
        uint256 indexed roundId,
        address indexed client,
        uint256 samples,
        uint256 score
    );

    /// MODIFIERS

    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }

    modifier onlyServer() {
        require(msg.sender == server, "Not server");
        _;
    }

    /// CONSTRUCTOR

    constructor(address _server) {
        require(_server != address(0), "Invalid server");
        owner = msg.sender;
        server = _server;
    }

    /// OWNER / ADMIN FUNCS

    function setServer(address _server) external onlyOwner {
        require(_server != address(0), "Invalid server");
        server = _server;
        emit ServerUpdated(_server);
    }

    function registerClient(address client) external onlyOwner {
        require(client != address(0), "Invalid client");
        require(!isClient[client], "Already registered");
        isClient[client] = true;
        emit ClientRegistered(client);
    }

    function removeClient(address client) external onlyOwner {
        require(isClient[client], "Not registered");
        isClient[client] = false;
        emit ClientRemoved(client);
    }

    /// MAIN SERVER FUNC

    /**
     * @notice Records a completed Federated Learning round.
     * @param roundId         The ID of the round (must be sequential)
     * @param modelHash       Hash of the global model weights
     * @param artifactCid     Filecoin/IPFS CID of the artifact bundle (model file)
     * @param summaryCid      Filecoin/IPFS CID of the summary data (for LLM queries)
     * @param clients         Participating client wallet addresses
     * @param samples         Per-client sample counts
     * @param scores          Per-client contribution scores
     */
    function recordRound(
        uint256 roundId,
        bytes32 modelHash,
        string calldata artifactCid,
        string calldata summaryCid,
        address[] calldata clients,
        uint256[] calldata samples,
        uint256[] calldata scores
    ) external onlyServer {

        require(roundId == latestRoundId + 1, "RoundId must increment sequentially");
        require(
            clients.length == samples.length && clients.length == scores.length,
            "Array length mismatch"
        );
        require(clients.length > 0, "No clients provided");
        require(bytes(artifactCid).length > 0, "Empty artifact CID");
        // summaryCid is optional (can be empty string if not available)

        // Validate clients and check for duplicates
        for (uint256 i = 0; i < clients.length; i++) {
            require(isClient[clients[i]], "Unregistered client");
            // Check for duplicates within this round
            for (uint256 j = i + 1; j < clients.length; j++) {
                require(clients[i] != clients[j], "Duplicate client in round");
            }
        }

        // Create round record
        RoundInfo storage r = rounds[roundId];
        r.modelHash = modelHash;
        r.artifactCid = artifactCid;
        r.summaryCid = summaryCid;
        r.timestamp = block.timestamp;

        // Store per-client contribution info
        for (uint256 i = 0; i < clients.length; i++) {
            address client = clients[i];
            r.clients.push(client);

            clientRounds[roundId][client] = ClientRoundInfo({
                samples: samples[i],
                score: scores[i]
            });

            emit ClientContributionRecorded(roundId, client, samples[i], scores[i]);
        }

        latestRoundId = roundId;

        emit RoundRecorded(
            roundId,
            modelHash,
            artifactCid,
            summaryCid,
            clients,
            block.timestamp
        );
    }
}
