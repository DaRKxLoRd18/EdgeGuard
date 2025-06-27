// const User = require('../models/User');

// exports.registerUser = async (req, res) => {
//   try {
//     const { name, email, location } = req.body;
//     const user = await User.create({ name, email, location });
//     res.status(201).json(user);
//   } catch (error) {
//     console.error("❌ Registration Error:", error.message);
//     res.status(400).json({ message: 'Error registering user', error: error.message });
//   }
// };


// exports.getUserByEmail = async (req, res) => {
//   try {
//     const { email } = req.query;
//     const user = await User.findOne({ email });

//     if (!user) {
//       return res.status(404).json({ message: "User not found" });
//     }

//     res.json(user);
//   } catch (error) {
//     console.error("Error fetching user by email:", error);
//     res.status(500).json({ message: "Server error" });
//   }
// };



// exports.getAllUsers = async (req, res) => {
//   const users = await User.find();
//   res.json(users);
// };

// exports.getUserById = async (req, res) => {
//   const user = await User.findById(req.params.id);
//   res.json(user);
// };

// exports.getUserWithHistory = async (req, res) => {
//   try {
//     const user = await User.findById(req.params.id)
//       .populate('alerts')
//       .populate('anomalies');
//     res.json(user);
//   } catch (err) {
//     res.status(404).json({ message: 'User not found', err });
//   }
// };



const User = require('../models/User');

exports.registerUser = async (req, res) => {
  try {
    const { name, email, location } = req.body;
    const user = await User.create({ name, email, location });
    res.status(201).json(user);
  } catch (error) {
    console.error("❌ Registration Error:", error.message);
    res.status(400).json({ message: 'Error registering user', error: error.message });
  }
};

exports.getUserByEmail = async (req, res) => {
  try {
    const { email } = req.query;

    if (!email) {
      return res.status(400).json({ message: "Missing email in query" });
    }

    console.log("📥 Finding user by email:", email);

    const user = await User.findOne({ email });

    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    res.status(200).json(user);
  } catch (error) {
    console.error("❌ Error fetching user by email:", error.message);
    res.status(500).json({ message: "Server error", error: error.message });
  }
};


exports.getAllUsers = async (req, res) => {
  try {
    const users = await User.find();
    res.json(users);
  } catch (error) {
    console.error("❌ Error fetching all users:", error.message);
    res.status(500).json({ message: "Server error", error: error.message });
  }
};

exports.getUserById = async (req, res) => {
  try {
    const user = await User.findById(req.params.id);

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    console.error("❌ Error fetching user by ID:", error.message);
    res.status(500).json({ message: "Server error", error: error.message });
  }
};

exports.getUserWithHistory = async (req, res) => {
  try {
    const user = await User.findById(req.params.id)
      .populate('alerts')
      .populate('anomalies');

    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    res.json(user);
  } catch (error) {
    console.error("❌ Error fetching user with history:", error.message);
    res.status(500).json({ message: 'Server error', error: error.message });
  }
};
